"""Diagnostics for experiment 01 (layer-scale injection).

Two hooks are provided:

``LayerScaleStatsHook``
    Every ``interval`` iterations records, for all 24 interaction layers, the
    value of ``alpha_l``, ``alpha_l``'s gradient norm, and the residual
    statistics ``||x_l||``, ``||delta_l||``, ``||alpha_l * delta_l||`` and
    ``||alpha_l * delta_l|| / (||x_l|| + 1e-6)``.  It also records the total,
    classification, mask and dice losses, the learning rate and the peak
    allocated GPU memory.  Everything is appended to a JSONL file so the
    analysis scripts can rebuild the per-layer curves without re-parsing logs.

``LayerScaleGuardHook``
    Enforces the automatic abort conditions of the protocol:

    * NaN/Inf in any ``alpha_l``, in the loss, or in the gradients;
    * ``|alpha_l| > 5`` for ``streak_iters`` consecutive iterations;
    * residual ratio ``> 1.0`` for ``streak_iters`` consecutive iterations;
    * any non-zero gradient on a frozen backbone parameter.

    When a condition trips, the hook writes ``ABORTED.json`` into the work dir
    and stops the training loop.  The run must then be recorded as failed rather
    than silently reported.

Both hooks are no-ops when the model was built without layer scaling, so the
same config can drive the S0 baseline.
"""

import json
import math
import os
import os.path as osp

import torch
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmseg.registry import HOOKS

from ..models.backbones.cloud_adapter import (
    ConvnextInteractiveModule,
    ScaledConvnextInteractiveModule,
)

# Index into the tensor produced by ``residual_stats``.
STAT_X_NORM, STAT_DELTA_NORM, STAT_SCALED_NORM, STAT_RATIO = 0, 1, 2, 3


def _unwrap_model(runner):
    model = runner.model
    if hasattr(model, "module"):
        model = model.module
    return model


def _logger(runner):
    """The runner's logger, with a fallback that always resolves."""
    for attr in ("mmengine_logger", "logger"):
        logger = getattr(runner, attr, None)
        if logger is not None:
            return logger
    return MMLogger.get_current_instance()


def _current_lr(runner):
    """Base learning rate of the first param group, or NaN."""
    try:
        optim_wrapper = runner.optim_wrapper
        # OptimWrapperDict is used by multi-optimizer configurations; the
        # experiment uses one wrapper, but handling both keeps the hook safe.
        if hasattr(optim_wrapper, "get_lr"):
            lrs = optim_wrapper.get_lr()
        else:
            lrs = runner.get_lr()
    except Exception:  # pragma: no cover - runner API differences
        return float("nan")
    if isinstance(lrs, (list, tuple)) and lrs:
        return float(lrs[0])
    if isinstance(lrs, dict) and lrs:
        value = next(iter(lrs.values()))
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        return float(value)
    return float("nan")


def iter_interactive_modules(model):
    """Yield ``(name, module)`` for every ConvNeXt interaction module."""
    for name, module in model.named_modules():
        if isinstance(module, ConvnextInteractiveModule):
            yield name, module


def iter_scaled_modules(model):
    """Yield ``(name, module)`` only for the layer-scaled variants."""
    for name, module in model.named_modules():
        if isinstance(module, ScaledConvnextInteractiveModule):
            yield name, module


class _LayerScaleHookMixin:
    """Shared module discovery and stats-collection toggling."""

    def __init__(self, collect_stats=True, **kwargs):
        super().__init__(**kwargs)
        self.collect_stats = collect_stats
        self.modules = []  # list of (name, module), resolved in before_train

    def _resolve(self, runner):
        model = _unwrap_model(runner)
        self.modules = list(iter_interactive_modules(model))

    def _set_collect(self, flag):
        for _, module in self.modules:
            module.collect_stats = flag and self.collect_stats

    def before_train(self, runner):
        self._resolve(runner)
        self._set_collect(True)

    def after_train(self, runner):
        self._set_collect(False)

    # Validation/test forwards must not pollute the training statistics, and
    # keeping the reductions off keeps inference latency measurements honest.
    def before_val(self, runner):
        self._set_collect(False)

    def after_val(self, runner):
        self._set_collect(True)

    def before_test(self, runner):
        self._set_collect(False)

    def after_test(self, runner):
        self._set_collect(True)


@HOOKS.register_module()
class LayerScaleStatsHook(_LayerScaleHookMixin, Hook):
    """Per-``interval`` layer-scale and residual-ratio logging (protocol §6)."""

    def __init__(
        self,
        interval=500,
        out_file="layer_scale_stats.jsonl",
        collect_stats=True,
    ):
        super().__init__(collect_stats=collect_stats)
        self.interval = interval
        self.out_file = out_file
        self._grad_norms = {}
        self._handles = []
        self._path = None

    # -- gradient capture -------------------------------------------------
    # Reading ``param.grad`` in ``after_train_iter`` is unreliable because the
    # optimizer wrapper may already have zeroed it. A backward hook on the
    # parameter itself fires at the right moment regardless of that ordering.
    def _register_grad_hooks(self):
        for _, module in self.modules:
            alpha = getattr(module, "alpha", None)
            if alpha is None or not alpha.requires_grad:
                continue
            key = id(alpha)

            def hook(grad, key=key):
                self._grad_norms[key] = torch.linalg.vector_norm(grad.detach())

            self._handles.append(alpha.register_hook(hook))

    def before_train(self, runner):
        super().before_train(runner)
        self._register_grad_hooks()

    def _remove_grad_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # -- logging ----------------------------------------------------------
    def _alpha_arrays(self):
        values, grad_norms = [], []
        for _, module in self.modules:
            alpha = getattr(module, "alpha", None)
            if alpha is None:
                # Unscaled baseline: report alpha == 1.0 so the curves stay
                # comparable across variants.
                values.append(torch.ones((), device="cpu"))
                grad_norms.append(float("nan"))
                continue
            values.append(alpha.detach().reshape(-1).cpu())
            grad = self._grad_norms.get(id(alpha))
            grad_norms.append(
                float(grad.item()) if grad is not None else float("nan")
            )
        return values, grad_norms

    def _stat_arrays(self):
        """Stack the per-layer residual statistics, materialising them once."""
        rows = []
        for _, module in self.modules:
            stats = getattr(module, "last_stats", None)
            if stats is None:
                rows.append(torch.full((4,), float("nan")))
            else:
                rows.append(stats.detach().float().cpu())
        if not rows:
            return torch.full((0, 4), float("nan"))
        return torch.stack(rows)

    def after_train_iter(
        self, runner, batch_idx, data_batch=None, outputs=None
    ):
        if self._path is None:
            self._path = osp.join(runner.work_dir, self.out_file)
            _logger(runner).info(
                f"LayerScaleStatsHook: writing layer statistics to {self._path}"
            )

        iteration = runner.iter + 1
        if iteration % self.interval != 0:
            return

        alphas, grad_norms = self._alpha_arrays()
        # A scalar alpha contributes a single number; a channel-wise alpha
        # contributes one per channel. Flatten so both shapes can be summarised.
        flat = (
            torch.cat([value.reshape(-1) for value in alphas]).float()
            if alphas
            else torch.zeros(0)
        )
        stats = self._stat_arrays()

        def _per_layer(column):
            if stats.numel() == 0:
                return []
            return [float(v) for v in stats[:, column].tolist()]

        record = {
            "iter": iteration,
            "lr": _current_lr(runner),
            "peak_mem_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else float("nan")
            ),
            "loss": self._loss_value(outputs, "loss"),
            "loss_cls": self._loss_value(outputs, "decode.loss_cls"),
            "loss_mask": self._loss_value(outputs, "decode.loss_mask"),
            "loss_dice": self._loss_value(outputs, "decode.loss_dice"),
            "num_layers": len(self.modules),
            "alpha": [float(v) for v in flat.tolist()],
            "alpha_mean": float(flat.mean()) if flat.numel() else float("nan"),
            "alpha_std": (
                float(flat.std(unbiased=False)) if flat.numel() else float("nan")
            ),
            "alpha_max_abs": (
                float(flat.abs().max()) if flat.numel() else float("nan")
            ),
            "alpha_grad_norm": grad_norms,
            "x_norm": _per_layer(STAT_X_NORM),
            "delta_norm": _per_layer(STAT_DELTA_NORM),
            "scaled_norm": _per_layer(STAT_SCALED_NORM),
            "residual_ratio": _per_layer(STAT_RATIO),
        }
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        _logger(runner).info(
            "layer_scale iter={iter} alpha_mean={am:.6f} alpha_std={asd:.6f} "
            "alpha_max_abs={ama:.6f} ratio_mean={rm:.6f} ratio_max={rx:.6f}".format(
                iter=iteration,
                am=record["alpha_mean"],
                asd=record["alpha_std"],
                ama=record["alpha_max_abs"],
                rm=(
                    sum(record["residual_ratio"]) / len(record["residual_ratio"])
                    if record["residual_ratio"]
                    else float("nan")
                ),
                rx=(
                    max(record["residual_ratio"])
                    if record["residual_ratio"]
                    else float("nan")
                ),
            )
        )

    def after_train(self, runner):
        self._remove_grad_hooks()
        super().after_train(runner)

    @staticmethod
    def _loss_value(outputs, key):
        if not isinstance(outputs, dict):
            return float("nan")
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        if isinstance(value, (int, float)):
            return float(value)
        return float("nan")


@HOOKS.register_module()
class LayerScaleGuardHook(_LayerScaleHookMixin, Hook):
    """Automatic abort conditions (protocol §6)."""

    def __init__(
        self,
        alpha_max=5.0,
        ratio_max=1.0,
        streak_iters=1000,
        frozen_check_interval=500,
        frozen_prefix="backbone.",
        frozen_exclude=("cloud_adapter",),
        collect_stats=True,
    ):
        super().__init__(collect_stats=collect_stats)
        self.alpha_max = alpha_max
        self.ratio_max = ratio_max
        self.streak_iters = streak_iters
        self.frozen_check_interval = frozen_check_interval
        self.frozen_prefix = frozen_prefix
        self.frozen_exclude = tuple(frozen_exclude)
        self.alpha_streak = 0
        self.ratio_streak = 0
        self.abort_reasons = []
        self.aborted = False
        self._grad_finite = {}
        self._grad_handles = []

    def before_train(self, runner):
        super().before_train(runner)
        model = _unwrap_model(runner)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            def hook(grad, name=name):
                # Keep the reduction on-device. All flags are materialised in
                # one synchronization in after_train_iter.
                self._grad_finite[name] = torch.isfinite(grad.detach()).all()

            self._grad_handles.append(param.register_hook(hook))

    def after_train(self, runner):
        for handle in self._grad_handles:
            handle.remove()
        self._grad_handles = []
        super().after_train(runner)

    # -- abort plumbing ---------------------------------------------------
    def _abort(self, runner, reason):
        if self.aborted:
            return
        self.aborted = True
        self.abort_reasons.append(reason)
        _logger(runner).error(f"LayerScaleGuardHook ABORT: {reason}")
        record = {
            "iter": runner.iter + 1,
            "reasons": list(self.abort_reasons),
            "alpha_streak": self.alpha_streak,
            "ratio_streak": self.ratio_streak,
        }
        with open(
            osp.join(runner.work_dir, "ABORTED.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(record, handle, indent=2)
        self._stop_training(runner)

    @staticmethod
    def _stop_training(runner):
        """Stop an mmengine loop without raising out of the training stack."""
        loop = getattr(runner, "train_loop", None)
        if loop is None:
            return
        for attr in ("_stop_training", "stop_training"):
            if hasattr(loop, attr):
                try:
                    setattr(loop, attr, True)
                except AttributeError:
                    pass

    # -- checks -----------------------------------------------------------
    def _frozen_violation(self, runner):
        """Non-zero gradient on a frozen backbone parameter."""
        model = _unwrap_model(runner)
        worst = None
        for name, param in model.named_parameters():
            if not name.startswith(self.frozen_prefix):
                continue
            if any(key in name for key in self.frozen_exclude):
                continue
            if param.requires_grad:
                continue
            if param.grad is None:
                continue
            if not torch.any(param.grad != 0):
                continue
            # Only materialise the magnitude once a violation is suspected.
            worst = (name, float(param.grad.detach().abs().max().cpu()))
            break
        return worst

    def after_train_iter(
        self, runner, batch_idx, data_batch=None, outputs=None
    ):
        if self.aborted:
            return
        iteration = runner.iter + 1

        # --- loss finiteness (from the values mmengine already computed) ---
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                    if value.numel() != 1:
                        continue
                    if not torch.isfinite(value).item():
                        self._abort(runner, f"non-finite {key} at iter {iteration}")
                        return
                elif isinstance(value, (int, float)) and not math.isfinite(value):
                    self._abort(runner, f"non-finite {key} at iter {iteration}")
                    return

        if self._grad_finite:
            names = list(self._grad_finite)
            flags = torch.stack([self._grad_finite[name] for name in names])
            if not bool(flags.all().item()):
                bad_flags = flags.detach().cpu().tolist()
                bad_names = [name for name, ok in zip(names, bad_flags) if not ok]
                self._grad_finite.clear()
                self._abort(
                    runner,
                    f"non-finite gradient at iter {iteration}: "
                    + ", ".join(bad_names[:8]),
                )
                return
            self._grad_finite.clear()

        if not self.modules:
            return

        # --- alpha finiteness, magnitude streak, residual-ratio streak ---
        # Everything is reduced on-device and materialised with a single sync.
        ok_flags = []
        alpha_max_abs = []
        ratio_max = []
        for _, module in self.modules:
            alpha = getattr(module, "alpha", None)
            if alpha is not None:
                detached = alpha.detach()
                ok_flags.append(torch.isfinite(detached).all())
                alpha_max_abs.append(detached.abs().max())
            stats = getattr(module, "last_stats", None)
            if stats is not None:
                ok_flags.append(torch.isfinite(stats).all())
                ratio_max.append(stats.detach()[STAT_RATIO])

        if ok_flags and not bool(torch.stack(ok_flags).all().item()):
            self._abort(runner, f"non-finite alpha/residual at iter {iteration}")
            return

        if alpha_max_abs:
            exceeded = bool(
                (torch.stack(alpha_max_abs) > self.alpha_max).any().item()
            )
            self.alpha_streak = self.alpha_streak + 1 if exceeded else 0
            if self.alpha_streak >= self.streak_iters:
                self._abort(
                    runner,
                    f"|alpha_l| > {self.alpha_max} for {self.streak_iters} "
                    f"consecutive iters (reached at iter {iteration})",
                )
                return

        if ratio_max:
            exceeded = bool((torch.stack(ratio_max) > self.ratio_max).any().item())
            self.ratio_streak = self.ratio_streak + 1 if exceeded else 0
            if self.ratio_streak >= self.streak_iters:
                self._abort(
                    runner,
                    f"residual ratio > {self.ratio_max} for {self.streak_iters} "
                    f"consecutive iters (reached at iter {iteration})",
                )
                return

        # --- frozen backbone -------------------------------------------------
        if iteration % self.frozen_check_interval == 0:
            violation = self._frozen_violation(runner)
            if violation is not None:
                name, magnitude = violation
                self._abort(
                    runner,
                    f"frozen parameter {name} received a non-zero gradient "
                    f"(max |grad| = {magnitude:.3e}) at iter {iteration}",
                )
