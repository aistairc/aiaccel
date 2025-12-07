from __future__ import annotations

from fnmatch import fnmatch
import logging
from pathlib import Path
import re

import torch

import lightning as lt

from aiaccel.torch.lightning import load_checkpoint

logger = logging.getLogger(__name__)


class LoadPretrainedCallback(lt.Callback):
    """Initialize a model from a pretrained checkpoint before training or validation.

    The callback loads weights from ``model_path`` once fitting or validation begins,
    matches finetune parameters to pretrained ones using glob-like patterns, and copies
    the matching weights into the finetune module before any optimization steps run.

    Args:
        model_path: Directory containing checkpoints saved by :func:`load_checkpoint`.
        target_patterns: Glob expressions that describe finetune parameters which
            should be initialized from pretrained weights.
        pattern_map: Optional mapping from finetune patterns to pretrained ones.
            Wildcards (``"*"``) are allowed and must appear the same number of times
            on both sides of the mapping.
        source_excludes: Optional pretrained-side glob patterns that should never be
            copied even when referenced by ``pattern_map``.
        target_excludes: Optional finetune-side glob patterns that should never be
            overwritten.
        config_name: Name of the checkpoint configuration to load.

    Example::

        callback = LoadPretrainedCallback(
            model_path=Path("pretrain_ckpt"),
            target_patterns=["detr_module.*"],
            pattern_map={"backbone.*": "visual_backbone.*"},
            source_excludes=["detr_module.heads.cls_head.*"],
            config_name="merged_config.yaml",
        )
        trainer = lt.Trainer(callbacks=[callback])
        trainer.fit(model)
    """

    def __init__(
        self,
        model_path: Path,
        target_patterns: list[str],
        pattern_map: dict[str, str] | None = None,
        source_excludes: list[str] | None = None,
        target_excludes: list[str] | None = None,
        config_name: str = "merged_config.yaml",
    ) -> None:
        super().__init__()

        # remember configuration about where to load and which config to use
        self.model_path = Path(model_path)
        self.config_name = config_name

        pattern_map = pattern_map or {}
        assert set(pattern_map) <= set(target_patterns)

        # build pattern dictionary used to match finetune parameters to pretrained ones
        pattern_dict = {ptn: ptn for ptn in target_patterns}
        pattern_dict.update(pattern_map)

        # remember exclusion filters for finetune and pretrained parameters
        self.source_excludes = source_excludes or []
        self.target_excludes = target_excludes or []

        # cache the derived mappings and bookkeeping flags
        self._ptn_dict = pattern_dict
        self._loaded = False

    @torch.no_grad()
    def on_fit_start(self, trainer: lt.Trainer, pl_module: lt.LightningModule) -> None:  # type: ignore[override]
        """Load pretrained weights and copy them into matching finetune parameters."""
        if self._loaded:
            return

        # load pretrained checkpoint and copy it to CPU tensors
        src_model, *_ = load_checkpoint(self.model_path, self.config_name, device="cpu")
        src_state_dict = {name: weight.cpu() for name, weight in src_model.state_dict().items()}
        dst_state_dict = dict(pl_module.state_dict())

        # iterate over each user-defined pattern rule
        for dst_ptn, src_ptn in self._ptn_dict.items():
            assert dst_ptn.count("*") == src_ptn.count("*")
            rgx_ptn = re.compile("^" + re.escape(dst_ptn).replace(r"\*", "(.*)") + "$")
            update_state: dict[str, torch.Tensor] = {}

            # look for finetune parameters matching the current rule
            for dst_name, dst_weight in dst_state_dict.items():
                match_ptn = rgx_ptn.fullmatch(dst_name)
                if not match_ptn:
                    continue
                if any(fnmatch(dst_name, ptn) for ptn in self.target_excludes):
                    continue

                groups = iter(match_ptn.groups())
                src_name = "".join(next(groups) if ch == "*" else ch for ch in src_ptn)

                # ensure we only pull parameters that are not excluded
                if any(fnmatch(src_name, ptn) for ptn in self.source_excludes):
                    continue

                # fetch pretrained tensor and check compatibility before scheduling update
                src_weight = src_state_dict.get(src_name)
                assert src_weight is not None, (
                    f"Pretrained key not found: pretrained['{src_name}'] (for finetune['{dst_name}'])."
                )
                assert src_weight.shape == dst_weight.shape

                update_state[dst_name] = src_weight

                logger.debug(f"Parameter '{dst_name}' initialized from '{src_name}' in checkpoint.")

            # apply the collected updates for this rule and mark them as assigned
            assert update_state, f"No parameters matched rule: '{dst_ptn}' -> '{src_ptn}'."
            pl_module.load_state_dict(update_state, strict=False)

            for dst_name in update_state:
                dst_state_dict.pop(dst_name)

        # prevent re-loading so weights are only imported once
        self._loaded = True

    def on_validation_start(self, trainer: lt.Trainer, pl_module: lt.LightningModule) -> None:  # type: ignore[override]
        """Ensure pretrained weights are loaded before running validation."""
        self.on_fit_start(trainer, pl_module)
