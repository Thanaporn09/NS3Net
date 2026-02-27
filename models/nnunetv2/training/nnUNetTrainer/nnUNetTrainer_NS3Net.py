from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from nnunetv2.nets.NS3Net import NS3Net


class nnUNetTrainer_NS3Net(nnUNetTrainerNoDeepSupervision):
    """
    nnUNet v2 trainer for NS3Net with auxiliary MoE losses:
      - moe_lb_loss (load balancing)
      - ood_consistency_loss (two-view routing consistency)
    EMA code intentionally removed (no shadow weights, no EMA validation swap).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # -------------------------
        # Optim / schedule
        # -------------------------
        self.initial_lr = 4e-4
        self.num_epochs = 300
        self.weight_decay = 3e-2

        self.grad_clip_main = 1.0
        self.grad_clip_moe = 0.5

        self.use_bf16 = False
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and (not self.use_bf16)),
        )

        # -------------------------
        # Aux scheduling
        # -------------------------
        self.aux_start_epoch = 15

        self.enable_moe_aux_loss = True
        self.moe_lb_weight_max = 1e-2
        self.moe_lb_warmup_epochs = 30

        self.enable_ood_consistency = True
        self.ood_cons_weight_max = 0.1
        self.ood_cons_warmup_epochs = 30
        self.ood_symmetric_kl = True
        self.gate_ref_use_full = False

        # augmented view
        self.supervise_aug_view = False
        self.aug_seg_weight = 0.1

        self.aug_gamma_prob = 0.3
        self.aug_gamma_range = (0.8, 1.2)
        self.aug_noise_prob = 0.3
        self.aug_noise_std = 0.02
        self.aug_contrast_prob = 0.3
        self.aug_contrast_range = (0.85, 1.15)
        self.aug_brightness_prob = 0.2
        self.aug_brightness_range = (-0.05, 0.05)

    def set_deep_supervision_enabled(self, enabled: bool):
        # no deep supervision in this trainer
        return

    # -------------------------
    # Loss
    # -------------------------
    def build_loss(self):
        # batch_dice=True stabilizes gradients for rare classes
        return DC_and_CE_loss(
            {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    # -------------------------
    # Network
    # -------------------------
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)

        if len(configuration_manager.patch_size) != 2:
            raise RuntimeError("nnUNetTrainer_NS3Net is intended for 2D only.")

        model = NS3Net(
            in_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            dims=(48, 96, 192, 384),
            depths=(1, 1, 1, 1),
            depths_decoder=(1, 1, 1),
            drop_path_rate=0.1,
            d_state=16,
            attn_drop=0.0,
            num_experts=4,
            top_k=3,
            lf_cutoff=0.25,
            lf_softness=0.25,
            window_overlap=0.6,
            gate_grid=(8, 8),
            gate_hidden_ratio=4,
            enc_spec_pool_ratios=(1, 1, 1, 1),
            dec_spec_pool_ratios=(1, 1, 1),
            lb_loss_coef=0.1,
            ood_consistency_coef=1.0,
            fuse_hidden_ratio=4,
            fuse_init_open_bias=1.0,
        )
        return model

    # -------------------------
    # Optimizer / LR scheduler
    # -------------------------
    def configure_optimizers(self):
        decay, no_decay = [], []

        for name, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            n = name.lower()

            # No weight decay for biases, norms, and temperature scalars
            if (
                n.endswith("bias")
                or ("norm" in n)
                or ("layernorm" in n)
                or (".bn" in n)
                or (".ln" in n)
                or ("temperature" in n)
            ):
                no_decay.append(p)
            else:
                decay.append(p)

        opt = AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.initial_lr,
            betas=(0.9, 0.999),
            eps=1e-5,
        )

        num_warmup_epochs = 15
        num_total_epochs = self.num_epochs

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < num_warmup_epochs:
                return float(current_epoch) / float(max(1, num_warmup_epochs))
            progress = float(current_epoch - num_warmup_epochs) / float(
                max(1, num_total_epochs - num_warmup_epochs)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        sch = LambdaLR(opt, lr_lambda)
        return opt, sch

    def _delayed_warmup(self, max_w: float, start_epoch: int, warmup_epochs: int) -> float:
        epoch = int(getattr(self, "current_epoch", 0))
        if epoch < start_epoch:
            return 0.0
        if warmup_epochs <= 0:
            return float(max_w)
        t = min(max((epoch - start_epoch) / float(warmup_epochs), 0.0), 1.0)
        t_cos = 0.5 * (1.0 - math.cos(math.pi * t))
        return float(max_w) * t_cos

    def _get_moe_lb_weight(self) -> float:
        if not self.enable_moe_aux_loss:
            return 0.0
        return self._delayed_warmup(self.moe_lb_weight_max, self.aux_start_epoch, self.moe_lb_warmup_epochs)

    def _get_ood_cons_weight(self) -> float:
        if not self.enable_ood_consistency:
            return 0.0
        return self._delayed_warmup(self.ood_cons_weight_max, self.aux_start_epoch, self.ood_cons_warmup_epochs)

    @torch.no_grad()
    def _make_aug_view(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()

        if torch.rand(1, device=x.device).item() < self.aug_gamma_prob:
            gmin, gmax = self.aug_gamma_range
            gamma = torch.empty(1, device=x.device).uniform_(gmin, gmax).item()
            y_min = y.amin(dim=(2, 3), keepdim=True)
            y_max = y.amax(dim=(2, 3), keepdim=True)
            denom = (y_max - y_min).clamp_min(1e-6)
            yn = (y - y_min) / denom
            yn = yn.clamp(0.0, 1.0).pow(gamma)
            y = yn * denom + y_min

        if torch.rand(1, device=x.device).item() < self.aug_contrast_prob:
            amin, amax = self.aug_contrast_range
            alpha = torch.empty(1, device=x.device).uniform_(amin, amax).item()
            mean = y.mean(dim=(2, 3), keepdim=True)
            y = mean + alpha * (y - mean)

        if torch.rand(1, device=x.device).item() < self.aug_brightness_prob:
            bmin, bmax = self.aug_brightness_range
            delta = torch.empty(1, device=x.device).uniform_(bmin, bmax).item()
            y = y + delta

        if torch.rand(1, device=x.device).item() < self.aug_noise_prob:
            y = y + self.aug_noise_std * torch.randn_like(y)

        return y

    @staticmethod
    def _replicate_list(x: Optional[torch.Tensor], n: int) -> List[Optional[torch.Tensor]]:
        return [x] * int(n)

    def _infer_stage_depths(self) -> Dict[str, int]:
        net = self.network
        depths: Dict[str, int] = {}

        for k in ["enc0", "enc1", "enc2", "enc3"]:
            stage = getattr(net, k, None)
            if stage is None or not hasattr(stage, "blocks"):
                raise RuntimeError(f"Cannot infer depth: network.{k} missing or has no .blocks")
            depths[k] = len(stage.blocks)

        for k in ["dec2", "dec1", "dec0"]:
            blk = getattr(net, k, None)
            if blk is None or not hasattr(blk, "stage") or not hasattr(blk.stage, "blocks"):
                raise RuntimeError(f"Cannot infer depth: network.{k} missing or has no .stage.blocks")
            depths[k] = len(blk.stage.blocks)

        return depths

    def _build_gate_ref_pack_from_aux(self, aux: Dict[str, Any], gate_ref_is_full: bool) -> Dict[str, Any]:
        key = "gate_probs_full" if gate_ref_is_full else "gate_probs_coarse"
        g = aux.get(key, None)
        depths = self._infer_stage_depths()
        return {
            "enc0": self._replicate_list(g, depths["enc0"]),
            "enc1": self._replicate_list(g, depths["enc1"]),
            "enc2": self._replicate_list(g, depths["enc2"]),
            "enc3": self._replicate_list(g, depths["enc3"]),
            "dec2": self._replicate_list(g, depths["dec2"]),
            "dec1": self._replicate_list(g, depths["dec1"]),
            "dec0": self._replicate_list(g, depths["dec0"]),
            "gate_ref_is_full": bool(gate_ref_is_full),
        }

    # -------------------------
    # Safe add
    # -------------------------
    @staticmethod
    def _safe_add(total: torch.Tensor, term: Optional[torch.Tensor], weight: float) -> torch.Tensor:
        if term is None or weight == 0.0:
            return total
        if not torch.isfinite(term).all():
            return total
        return total + (weight * term)

    # -------------------------
    # Train step 
    # -------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        use_amp = (self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and self.use_bf16) else torch.float16
        use_scaler = bool(
            use_amp
            and (not self.use_bf16)
            and self.grad_scaler is not None
            and self.grad_scaler.is_enabled()
        )

        w_lb = float(self._get_moe_lb_weight())
        w_ood = float(self._get_ood_cons_weight())
        need_aux = (w_lb > 0.0) or (w_ood > 0.0)

        out: Dict[str, float] = {
            "loss": float("nan"),
            "seg_loss": float("nan"),
            "moe_lb_weight": w_lb,
            "ood_cons_weight": w_ood,
            "skipped_step": 0.0,
        }

        with autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
            if not need_aux:
                logits = self.network(data)
                seg_loss1 = self.loss(logits, target)
                total_loss = seg_loss1
            else:
                logits1, aux1 = self.network(data, return_aux=True)
                seg_loss1 = self.loss(logits1, target)
                total_loss = seg_loss1

                total_loss = self._safe_add(total_loss, aux1.get("moe_lb_loss", None), w_lb)

                data2 = self._make_aug_view(data)
                gate_ref_pack = self._build_gate_ref_pack_from_aux(aux1, gate_ref_is_full=self.gate_ref_use_full)
                logits2, aux2 = self.network(data2, return_aux=True, gate_ref_pack=gate_ref_pack)

                total_loss = self._safe_add(total_loss, aux2.get("moe_lb_loss", None), w_lb)
                total_loss = self._safe_add(total_loss, aux2.get("ood_consistency_loss", None), w_ood)

                if self.supervise_aug_view:
                    seg_loss2 = self.loss(logits2, target)
                    total_loss = total_loss + float(self.aug_seg_weight) * seg_loss2

                if self.ood_symmetric_kl and w_ood > 0.0:
                    gate_ref_pack_v2 = self._build_gate_ref_pack_from_aux(aux2, gate_ref_is_full=self.gate_ref_use_full)
                    _, aux1_sym = self.network(data, return_aux=True, gate_ref_pack=gate_ref_pack_v2)
                    total_loss = self._safe_add(total_loss, aux1_sym.get("ood_consistency_loss", None), w_ood * 0.5)

        # skip on non-finite loss
        if not torch.isfinite(total_loss).item():
            out["loss"] = float(total_loss.detach().cpu())
            out["seg_loss"] = float(seg_loss1.detach().cpu())
            out["skipped_step"] = 1.0
            self.optimizer.zero_grad(set_to_none=True)
            return out

        # backward
        if use_scaler:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()

        # grad clipping: separate MoE-ish params vs others
        moe_params = [
            p for n, p in self.network.named_parameters()
            if p.grad is not None and ("gate" in n.lower() or "expert" in n.lower())
        ]
        other_params = [
            p for n, p in self.network.named_parameters()
            if p.grad is not None and ("gate" not in n.lower() and "expert" not in n.lower())
        ]

        if moe_params:
            torch.nn.utils.clip_grad_norm_(moe_params, self.grad_clip_moe)
        grad_norm = torch.nn.utils.clip_grad_norm_(other_params, self.grad_clip_main)

        # skip step if grads become non-finite
        found_inf = not torch.isfinite(torch.as_tensor(grad_norm, device=self.device)).item()
        if not found_inf:
            for p in self.network.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    found_inf = True
                    break

        if found_inf:
            out["loss"] = float(total_loss.detach().cpu())
            out["seg_loss"] = float(seg_loss1.detach().cpu())
            out["skipped_step"] = 1.0
            self.optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                self.grad_scaler.update()
            return out

        # optimizer step
        if use_scaler:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        out["loss"] = float(total_loss.detach().cpu())
        out["seg_loss"] = float(seg_loss1.detach().cpu())
        return out

    # -------------------------
    # Validation step
    # -------------------------
    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        use_amp = (self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and self.use_bf16) else torch.float16

        with torch.no_grad(), autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
            logits = self.network(data)
            loss = self.loss(logits, target)

        axes = [0] + list(range(2, logits.ndim))

        # region-based labels (sigmoid)
        if self.label_manager.has_regions:
            t = (target[0] if isinstance(target, list) else target).float()
            pred = (torch.sigmoid(logits) > 0.5).float()

            if self.label_manager.has_ignore_label:
                ignore = t[:, -1:]
                mask = 1.0 - ignore
                t_eval = t[:, :-1]
                pred_eval = pred[:, :-1]
            else:
                mask, t_eval, pred_eval = None, t, pred

            tp, fp, fn, _ = get_tp_fp_fn_tn(pred_eval, t_eval, axes=axes, mask=mask)
            return {
                "loss": loss.detach().cpu().numpy(),
                "tp_hard": tp.detach().cpu().numpy(),
                "fp_hard": fp.detach().cpu().numpy(),
                "fn_hard": fn.detach().cpu().numpy(),
            }

        # multiclass: argmax → one-hot
        pred_lbl = logits.argmax(1)[:, None]
        pred_oh = torch.zeros_like(logits, dtype=torch.float32)
        pred_oh.scatter_(1, pred_lbl, 1.0)

        t0 = (target[0] if isinstance(target, list) else target)
        if t0.ndim == logits.ndim:
            tgt_oh = t0.float()
        else:
            lbl = t0[:, 0].long()
            tgt_oh = torch.zeros_like(logits, dtype=torch.float32)
            tgt_oh.scatter_(1, lbl[:, None], 1.0)

        mask = ((t0 != self.label_manager.ignore_label).float() if self.label_manager.has_ignore_label else None)
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_oh, tgt_oh, axes=axes, mask=mask)

        tp_h = tp.detach().cpu().numpy()
        fp_h = fp.detach().cpu().numpy()
        fn_h = fn.detach().cpu().numpy()

        # drop background channel if present
        if tp_h.shape[0] > 1:
            tp_h = tp_h[1:]
            fp_h = fp_h[1:]
            fn_h = fn_h[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_h,
            "fp_hard": fp_h,
            "fn_hard": fn_h,
        }