from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from nnunetv2.nets.ss2d import SS2D  # must exist in your repo

# nicer repr in logs
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# =============================================================================
# Layout helpers
# =============================================================================
class NCHWtoNHWC(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()


class NHWCtoNCHW(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()


# =============================================================================
# Soft radial windows cache (rFFT grid)
# =============================================================================
class SoftRadialWindowCache:
    """
    Cache N soft/overlapping raised-cosine radial windows for rFFT2 grids.

    For rFFT2:
      fy from fftfreq(H)   → [-0.5, 0.5)
      fx from rfftfreq(W)  → [0, 0.5]
    radial is normalized to [0, 1].

    Returns: List[Tensor], each [1, 1, H, Wf] where Wf = W//2 + 1.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple, List[torch.Tensor]] = {}

    @torch.no_grad()
    def get_windows(
        self,
        H: int,
        W: int,
        num_bands: int,
        overlap: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        key = (H, W, num_bands, float(overlap), device, dtype)
        if key in self._cache:
            return self._cache[key]

        Wf = W // 2 + 1
        fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=torch.float32)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")  # [H, Wf]
        radial = torch.sqrt(yy * yy + xx * xx)
        max_r = radial.max().clamp_min(1e-8)
        r = (radial / max_r).unsqueeze(0).unsqueeze(0)  # [1,1,H,Wf] in [0,1]

        centers = torch.linspace(0.0, 1.0, num_bands, device=device, dtype=torch.float32)
        base_hw = 0.5 / max(num_bands - 1, 1) if num_bands > 1 else 0.5
        half_width = base_hw * (1.0 + float(overlap))

        windows: List[torch.Tensor] = []
        denom = max(float(half_width), 1e-6)
        for c in centers:
            dist = (r - c).abs()
            w = torch.zeros_like(r)
            inside = dist <= half_width
            w[inside] = 0.5 * (1.0 + torch.cos(math.pi * (dist[inside] / denom)))
            windows.append(w.to(dtype=dtype))

        self._cache[key] = windows
        return windows


_WINDOW_CACHE = SoftRadialWindowCache()


def _soft_lowpass_mask(
    H: int,
    W: int,
    cutoff: float,
    softness: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Smooth low-pass mask on rFFT grid: [1, 1, H, Wf], values in [0, 1].
    cutoff and softness in (0, 1].
    """
    fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radial = torch.sqrt(yy * yy + xx * xx)
    max_r = radial.max().clamp_min(1e-8)
    r = (radial / max_r).unsqueeze(0).unsqueeze(0)  # [1,1,H,Wf]

    cutoff = float(cutoff)
    softness = float(softness)
    t = max(cutoff * softness, 1e-4)

    lp = torch.zeros_like(r)
    lp[r <= (cutoff - t)] = 1.0
    lp[r >= (cutoff + t)] = 0.0

    mid = (r > (cutoff - t)) & (r < (cutoff + t))
    u = (r[mid] - (cutoff - t)) / (2.0 * t)
    lp[mid] = 0.5 * (1.0 + torch.cos(math.pi * u))  # 1 → 0

    return lp.to(dtype=dtype)


# =============================================================================
# Region-wise MoE gate
# =============================================================================
class RegionUniversalGate(nn.Module):
    """
    Region-wise routing probabilities over N experts.

    Input:  NCHW [B, C, Hf, Wf]
    Output: gate_probs [B, N, hg, wg] (softmax over experts; top-k sparse per cell)
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        top_k: int = 1,
        gate_grid: Tuple[int, int] = (8, 8),
        gate_hidden_ratio: int = 4,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.gate_grid = gate_grid

        hidden = max(in_channels // gate_hidden_ratio, 16)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.to_logits = nn.Conv2d(hidden, num_experts, 1, bias=True)
        self.temperature = nn.Parameter(torch.ones(1))
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hg, wg = self.gate_grid
        x_ds = F.adaptive_avg_pool2d(x, (hg, wg))
        feat = self.backbone(x_ds)
        logits = self.to_logits(feat) / (self.temperature + self.eps)

        topk_logits, idx = torch.topk(logits, k=self.top_k, dim=1)
        masked = torch.full_like(logits, -float("inf"))
        masked.scatter_(1, idx, topk_logits)
        return F.softmax(masked, dim=1)


# =============================================================================
# Spectral experts (amplitude-domain)
# =============================================================================
class SpectralAmpExpert(nn.Module):
    def __init__(self, C: int, hidden_ratio: float = 0.5) -> None:
        super().__init__()
        Ch = max(int(C * hidden_ratio), 16)
        self.net = nn.Sequential(
            nn.Conv2d(C, Ch, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(Ch, C, 1, bias=True),
        )
        self.act = nn.GELU()

    def forward(self, amp: torch.Tensor) -> torch.Tensor:
        return amp + self.act(self.net(amp))


# =============================================================================
# NS3MoE: region-wise frequency-routed MoE on rFFT magnitudes (phase preserved)
# =============================================================================
class NS3MoE(nn.Module):
    """
    NS3MoE (manuscript name):
      - rFFT2 → split amplitude into low/high frequency via smooth low-pass
      - high-frequency magnitude routed to spectral experts with region-wise gate
      - preserve phase; reconstruct via irFFT2
      - optionally returns aux for load-balance and routing-consistency losses

    Input/Output layout: NHWC (to match SS2D blocks).
    """

    def __init__(
        self,
        C: int,
        num_experts: int = 4,
        top_k: int = 1,
        lf_cutoff: float = 0.25,
        lf_softness: float = 0.25,
        window_overlap: float = 0.5,
        gate_grid: Tuple[int, int] = (8, 8),
        gate_hidden_ratio: int = 4,
        spec_pool_ratio: int = 1,
        lb_loss_coef: float = 0.01,           
        ood_consistency_coef: float = 0.05,   
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        assert spec_pool_ratio >= 1
        assert 0.0 < lf_cutoff < 1.0
        assert 0.0 < lf_softness <= 1.0
        assert 0.0 <= window_overlap <= 1.0

        self.C = int(C)
        self.N = int(num_experts)
        self.top_k = int(top_k)
        self.lf_cutoff = float(lf_cutoff)
        self.lf_softness = float(lf_softness)
        self.window_overlap = float(window_overlap)
        self.gate_grid = gate_grid
        self.spec_pool_ratio = int(spec_pool_ratio)
        self.eps = float(eps)

        # kept for compatibility/logging
        self.lb_loss_coef = float(lb_loss_coef)
        self.ood_consistency_coef = float(ood_consistency_coef)

        self.gate_in = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=True),
            nn.GELU(),
        )
        self.gate = RegionUniversalGate(
            in_channels=C,
            num_experts=self.N,
            top_k=self.top_k,
            gate_grid=self.gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
        )
        self.experts = nn.ModuleList([SpectralAmpExpert(C=C, hidden_ratio=0.5) for _ in range(self.N)])

        # small post-fusion refinement in amplitude domain
        self.amp_refine = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(C, C, 1, bias=True),
        )

    def _maybe_spec_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        Hf, Wf = x.shape[-2:]
        if self.spec_pool_ratio == 1:
            return x, (Hf, Wf)
        r = self.spec_pool_ratio
        x_ds = F.avg_pool2d(x, kernel_size=r, stride=r, ceil_mode=False)
        return x_ds, (Hf, Wf)

    def _spec_unpool(self, y: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if y.shape[-2:] == target_hw:
            return y
        return F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)

    def _load_balance_loss(self, gate_probs_full: torch.Tensor) -> torch.Tensor:
        p = gate_probs_full.mean(dim=(0, 2, 3))  # [N]
        uniform = torch.full_like(p, 1.0 / p.numel())
        return F.mse_loss(p, uniform)

    def _gate_entropy(self, gate_probs_full: torch.Tensor) -> torch.Tensor:
        p = gate_probs_full.clamp_min(self.eps)
        ent = -(p * p.log()).sum(dim=1)  # [B,H,W]
        return ent.mean()

    def _ood_consistency_loss(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(self.eps)
        q = q.clamp_min(self.eps)
        kl = (p * (p.log() - q.log())).sum(dim=1)  # [B,H,W]
        return kl.mean()

    def forward(
        self,
        x: torch.Tensor,  # NHWC
        return_aux: bool = False,
        gate_ref: Optional[torch.Tensor] = None,
        gate_ref_is_full: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        B, H, W, C = x.shape
        if C != self.C:
            raise RuntimeError(f"Channel mismatch: got {C}, expected {self.C}")

        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        orig_dtype = x_nchw.dtype

        # IMPORTANT: keep FFT in fp32 to avoid numerical issues; do not autocast.
        x_fp32 = x_nchw.float()

        # rFFT2: complex64
        X = torch.fft.rfft2(x_fp32)
        amp = torch.abs(X)          # magnitude
        phase = torch.angle(X)      # preserve phase
        Hf, Wf = amp.shape[-2], amp.shape[-1]

        # smooth low-pass split on rFFT grid (needs original W)
        lp = _soft_lowpass_mask(
            H=Hf,
            W=(Wf - 1) * 2,
            cutoff=self.lf_cutoff,
            softness=self.lf_softness,
            device=amp.device,
            dtype=amp.dtype,
        )
        amp_lf = amp * lp
        amp_hf = amp * (1.0 - lp)

        # optional spectral pooling for cheaper gating/experts
        amp_hf_ds, target_hw = self._maybe_spec_pool(amp_hf)
        Hds, Wds = amp_hf_ds.shape[-2:]

        # per-band radial windows on the (downsampled) rFFT grid
        W_approx = (Wds - 1) * 2
        windows = _WINDOW_CACHE.get_windows(
            H=Hds,
            W=W_approx,
            num_bands=self.N,
            overlap=self.window_overlap,
            device=amp_hf_ds.device,
            dtype=amp_hf_ds.dtype,
        )

        # region-wise gate from log-amplitude
        gate_feat = self.gate_in(torch.log1p(amp_hf_ds + self.eps))
        gate_probs_coarse = self.gate(gate_feat)  # [B,N,hg,wg]

        gate_probs_full = F.interpolate(
            gate_probs_coarse, size=(Hds, Wds), mode="bilinear", align_corners=False
        )
        gate_probs_full = gate_probs_full.clamp_min(self.eps)
        gate_probs_full = gate_probs_full / gate_probs_full.sum(dim=1, keepdim=True)

        # route each radial band through its expert, mix by gate
        out_hf = torch.zeros_like(amp_hf_ds)
        for i, (expert, win) in enumerate(zip(self.experts, windows)):
            band = amp_hf_ds * win
            eo = expert(band)
            g = gate_probs_full[:, i : i + 1, :, :]
            out_hf = out_hf + eo * g

        out_hf = out_hf + self.amp_refine(out_hf)
        out_hf_full = self._spec_unpool(out_hf, target_hw=target_hw)

        # reconstruct magnitude (phase preserved)
        amp_rec = (amp_lf + out_hf_full).clamp_min(0.0)
        X_rec = torch.polar(amp_rec, phase)
        delta = torch.fft.irfft2(X_rec, s=(H, W))  # [B,C,H,W], fp32

        delta = delta.to(orig_dtype)
        delta_nhwc = delta.permute(0, 2, 3, 1).contiguous()

        if not return_aux:
            return delta_nhwc

        # aux computed on the full rFFT grid resolution
        gate_full_up = F.interpolate(
            gate_probs_full, size=(Hf, Wf), mode="bilinear", align_corners=False
        )
        gate_full_up = gate_full_up.clamp_min(self.eps)
        gate_full_up = gate_full_up / gate_full_up.sum(dim=1, keepdim=True)

        lb_raw = self._load_balance_loss(gate_full_up)
        ent = self._gate_entropy(gate_full_up)
        usage = gate_full_up.mean(dim=(0, 2, 3))

        ood_raw: Optional[torch.Tensor] = None
        if gate_ref is not None:
            if gate_ref_is_full:
                gate_ref_full = gate_ref
            else:
                gate_ref_full = F.interpolate(
                    gate_ref, size=(Hf, Wf), mode="bilinear", align_corners=False
                )
            gate_ref_full = gate_ref_full.clamp_min(self.eps)
            gate_ref_full = gate_ref_full / gate_ref_full.sum(dim=1, keepdim=True)
            ood_raw = self._ood_consistency_loss(gate_full_up, gate_ref_full)

        aux: Dict[str, Any] = {
            "moe_lb_loss": lb_raw,
            "gate_entropy": ent.detach(),
            "expert_usage": usage.detach(),
            "gate_probs_coarse": gate_probs_coarse.detach(),
            "gate_probs_full": gate_full_up.detach(),
        }
        if ood_raw is not None:
            aux["ood_consistency_loss"] = ood_raw

        return delta_nhwc, aux


# =============================================================================
# NS3Block: SS2D + NS3MoE (manuscript name)
# =============================================================================
class NS3Block(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        d_state: int = 16,
        num_experts: int = 4,
        top_k: int = 1,
        lf_cutoff: float = 0.25,
        lf_softness: float = 0.25,
        window_overlap: float = 0.5,
        gate_grid: Tuple[int, int] = (8, 8),
        gate_hidden_ratio: int = 4,
        spec_pool_ratio: int = 1,
        lb_loss_coef: float = 0.01,
        ood_consistency_coef: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path)
        self.ssm = SS2D(d_model=dim, dropout=attn_drop, d_state=d_state, **kwargs)

        self.norm2 = norm_layer(dim)
        self.moe = NS3MoE(
            C=dim,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=spec_pool_ratio,
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
        )

    def forward(
        self,
        x: torch.Tensor,  # NHWC
        return_aux: bool = False,
        gate_ref: Optional[torch.Tensor] = None,
        gate_ref_is_full: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        x = x + self.drop_path(self.ssm(self.norm1(x)))

        if not return_aux:
            delta = self.moe(self.norm2(x), return_aux=False)
            x = x + self.drop_path(delta)
            return x

        delta, aux = self.moe(
            self.norm2(x),
            return_aux=True,
            gate_ref=gate_ref,
            gate_ref_is_full=gate_ref_is_full,
        )
        x = x + self.drop_path(delta)
        return x, aux


# =============================================================================
# Stage wrapper (NCHW external; NHWC inside) with aux aggregation
# =============================================================================
class NS3StageNCHW(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_seq: List[float],
        norm_layer=nn.LayerNorm,
        d_state: int = 16,
        attn_drop: float = 0.0,
        num_experts: int = 4,
        top_k: int = 1,
        lf_cutoff: float = 0.25,
        lf_softness: float = 0.25,
        window_overlap: float = 0.5,
        gate_grid: Tuple[int, int] = (8, 8),
        gate_hidden_ratio: int = 4,
        spec_pool_ratio: int = 1,
        lb_loss_coef: float = 0.01,
        ood_consistency_coef: float = 0.05,
    ) -> None:
        super().__init__()
        if len(drop_path_seq) != depth:
            raise ValueError("drop_path_seq length must match depth")

        self.to_nhwc = NCHWtoNHWC()
        self.to_nchw = NHWCtoNCHW()

        self.blocks = nn.ModuleList(
            [
                NS3Block(
                    dim=dim,
                    drop_path=drop_path_seq[i],
                    norm_layer=norm_layer,
                    d_state=d_state,
                    attn_drop=attn_drop,
                    num_experts=num_experts,
                    top_k=top_k,
                    lf_cutoff=lf_cutoff,
                    lf_softness=lf_softness,
                    window_overlap=window_overlap,
                    gate_grid=gate_grid,
                    gate_hidden_ratio=gate_hidden_ratio,
                    spec_pool_ratio=spec_pool_ratio,
                    lb_loss_coef=lb_loss_coef,
                    ood_consistency_coef=ood_consistency_coef,
                )
                for i in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,  # NCHW
        return_aux: bool = False,
        gate_ref_list: Optional[List[Optional[torch.Tensor]]] = None,
        gate_ref_is_full: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        x = self.to_nhwc(x)

        if not return_aux:
            for b in self.blocks:
                x = b(x, return_aux=False)
            return self.to_nchw(x)

        if gate_ref_list is None:
            gate_ref_list = [None] * len(self.blocks)

        moe_lb: Optional[torch.Tensor] = None
        ood_cons: Optional[torch.Tensor] = None
        gate_entropy_sum: Optional[torch.Tensor] = None
        expert_usage_sum: Optional[torch.Tensor] = None
        last_gate_coarse: Optional[torch.Tensor] = None
        last_gate_full: Optional[torch.Tensor] = None
        n_blocks = len(self.blocks)

        for b, gref in zip(self.blocks, gate_ref_list):
            x, aux = b(x, return_aux=True, gate_ref=gref, gate_ref_is_full=gate_ref_is_full)

            moe_lb = aux["moe_lb_loss"] if moe_lb is None else (moe_lb + aux["moe_lb_loss"])

            ge = aux.get("gate_entropy", None)
            if ge is not None:
                gate_entropy_sum = ge if gate_entropy_sum is None else (gate_entropy_sum + ge)

            eu = aux.get("expert_usage", None)
            if eu is not None:
                expert_usage_sum = eu if expert_usage_sum is None else (expert_usage_sum + eu)

            last_gate_coarse = aux["gate_probs_coarse"]
            last_gate_full = aux["gate_probs_full"]

            if "ood_consistency_loss" in aux:
                ood_cons = aux["ood_consistency_loss"] if ood_cons is None else (ood_cons + aux["ood_consistency_loss"])

        x = self.to_nchw(x)

        out_aux: Dict[str, Any] = {
            "moe_lb_loss": moe_lb,
            "gate_entropy": (gate_entropy_sum / n_blocks).detach() if gate_entropy_sum is not None else None,
            "expert_usage": (expert_usage_sum / n_blocks).detach() if expert_usage_sum is not None else None,
            "gate_probs_coarse": last_gate_coarse,
            "gate_probs_full": last_gate_full,
        }
        if ood_cons is not None:
            out_aux["ood_consistency_loss"] = ood_cons

        return x, out_aux


# =============================================================================
# Down / Up blocks
# =============================================================================
class Down2x(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up2x(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.proj(x)


# =============================================================================
# Gated skip fusion
# =============================================================================
class GatedSkipFuse(nn.Module):
    """
    Per-channel gated additive skip connection:
        y = x_up + sigmoid(g(x_up, skip)) * proj(skip)
    Gate uses global-pooled descriptors for spatial stability.
    """

    def __init__(
        self,
        up_ch: int,
        skip_ch: int,
        out_ch: int,
        gate_hidden_ratio: int = 4,
        init_open_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.proj_up = nn.Identity() if up_ch == out_ch else nn.Conv2d(up_ch, out_ch, 1, bias=False)
        self.proj_skip = nn.Conv2d(skip_ch, out_ch, 1, bias=False)

        hidden = max(out_ch // gate_hidden_ratio, 16)
        self.gate_fc1 = nn.Conv2d(out_ch * 2, hidden, 1, bias=True)
        self.gate_act = nn.GELU()
        self.gate_fc2 = nn.Conv2d(hidden, out_ch, 1, bias=True)
        self.gate_sig = nn.Sigmoid()

        nn.init.constant_(self.gate_fc2.bias, init_open_bias)

    def forward(self, x_up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.proj_up(x_up)
        s = self.proj_skip(skip)

        xu = F.adaptive_avg_pool2d(x_up, 1)
        ss = F.adaptive_avg_pool2d(s, 1)
        g = self.gate_sig(self.gate_fc2(self.gate_act(self.gate_fc1(torch.cat([xu, ss], dim=1)))))

        return x_up + g * s


# =============================================================================
# Decoder block
# =============================================================================
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        depth: int,
        drop_path_seq: List[float],
        norm_layer=nn.LayerNorm,
        d_state: int = 16,
        attn_drop: float = 0.0,
        num_experts: int = 4,
        top_k: int = 1,
        lf_cutoff: float = 0.25,
        lf_softness: float = 0.25,
        window_overlap: float = 0.5,
        gate_grid: Tuple[int, int] = (8, 8),
        gate_hidden_ratio: int = 4,
        spec_pool_ratio: int = 1,
        lb_loss_coef: float = 0.01,
        ood_consistency_coef: float = 0.05,
        fuse_hidden_ratio: int = 4,
        fuse_init_open_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.up = Up2x(in_ch, out_ch)
        self.fuse = GatedSkipFuse(
            up_ch=out_ch,
            skip_ch=skip_ch,
            out_ch=out_ch,
            gate_hidden_ratio=fuse_hidden_ratio,
            init_open_bias=fuse_init_open_bias,
        )
        self.stage = NS3StageNCHW(
            dim=out_ch,
            depth=depth,
            drop_path_seq=drop_path_seq,
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=spec_pool_ratio,
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        return_aux: bool = False,
        gate_ref_list: Optional[List[Optional[torch.Tensor]]] = None,
        gate_ref_is_full: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        x = self.up(x, target_hw=skip.shape[-2:])
        x = self.fuse(x, skip)

        if not return_aux:
            return self.stage(x, return_aux=False)

        x, aux = self.stage(x, return_aux=True, gate_ref_list=gate_ref_list, gate_ref_is_full=gate_ref_is_full)
        return x, aux

class NS3Net(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        depths: Tuple[int, int, int, int] = (1, 1, 1, 1),
        dims: Tuple[int, int, int, int] = (48, 96, 192, 384),
        depths_decoder: Tuple[int, int, int] = (1, 1, 1),
        norm_layer=nn.LayerNorm,
        drop_path_rate: float = 0.15,
        d_state: int = 16,
        attn_drop: float = 0.0,
        num_experts: int = 4,
        top_k: int = 2,
        lf_cutoff: float = 0.25,
        lf_softness: float = 0.25,
        window_overlap: float = 0.6,
        gate_grid: Tuple[int, int] = (8, 8),
        gate_hidden_ratio: int = 4,
        enc_spec_pool_ratios: Tuple[int, int, int, int] = (1, 1, 1, 2),
        dec_spec_pool_ratios: Tuple[int, int, int] = (1, 1, 1),
        lb_loss_coef: float = 1.0,
        ood_consistency_coef: float = 1.0,
        fuse_hidden_ratio: int = 4,
        fuse_init_open_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if not (len(depths) == 4 and len(dims) == 4 and len(enc_spec_pool_ratios) == 4):
            raise ValueError("depths/dims/enc_spec_pool_ratios must have length 4")
        if not (len(depths_decoder) == 3 and len(dec_spec_pool_ratios) == 3):
            raise ValueError("depths_decoder/dec_spec_pool_ratios must have length 3")

        # stem downsamples by 4×
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, padding=0, bias=True)

        enc_dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        p = 0

        self.enc0 = NS3StageNCHW(
            dim=dims[0],
            depth=depths[0],
            drop_path_seq=enc_dpr[p : p + depths[0]],
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=enc_spec_pool_ratios[0],
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
        )
        p += depths[0]
        self.down0 = Down2x(dims[0], dims[1])

        self.enc1 = NS3StageNCHW(
            dim=dims[1],
            depth=depths[1],
            drop_path_seq=enc_dpr[p : p + depths[1]],
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=enc_spec_pool_ratios[1],
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
        )
        p += depths[1]
        self.down1 = Down2x(dims[1], dims[2])

        self.enc2 = NS3StageNCHW(
            dim=dims[2],
            depth=depths[2],
            drop_path_seq=enc_dpr[p : p + depths[2]],
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=enc_spec_pool_ratios[2],
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
        )
        p += depths[2]
        self.down2 = Down2x(dims[2], dims[3])

        self.enc3 = NS3StageNCHW(
            dim=dims[3],
            depth=depths[3],
            drop_path_seq=enc_dpr[p : p + depths[3]],
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=enc_spec_pool_ratios[3],
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
        )

        dec_dpr = torch.linspace(0, drop_path_rate, sum(depths_decoder)).tolist()
        q = 0

        self.dec2 = DecoderBlock(
            in_ch=dims[3],
            skip_ch=dims[2],
            out_ch=dims[2],
            depth=depths_decoder[0],
            drop_path_seq=dec_dpr[q : q + depths_decoder[0]],
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=dec_spec_pool_ratios[0],
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
            fuse_hidden_ratio=fuse_hidden_ratio,
            fuse_init_open_bias=fuse_init_open_bias,
        )
        q += depths_decoder[0]

        self.dec1 = DecoderBlock(
            in_ch=dims[2],
            skip_ch=dims[1],
            out_ch=dims[1],
            depth=depths_decoder[1],
            drop_path_seq=dec_dpr[q : q + depths_decoder[1]],
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=dec_spec_pool_ratios[1],
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
            fuse_hidden_ratio=fuse_hidden_ratio,
            fuse_init_open_bias=fuse_init_open_bias,
        )
        q += depths_decoder[1]

        self.dec0 = DecoderBlock(
            in_ch=dims[1],
            skip_ch=dims[0],
            out_ch=dims[0],
            depth=depths_decoder[2],
            drop_path_seq=dec_dpr[q : q + depths_decoder[2]],
            norm_layer=norm_layer,
            d_state=d_state,
            attn_drop=attn_drop,
            num_experts=num_experts,
            top_k=top_k,
            lf_cutoff=lf_cutoff,
            lf_softness=lf_softness,
            window_overlap=window_overlap,
            gate_grid=gate_grid,
            gate_hidden_ratio=gate_hidden_ratio,
            spec_pool_ratio=dec_spec_pool_ratios[2],
            lb_loss_coef=lb_loss_coef,
            ood_consistency_coef=ood_consistency_coef,
            fuse_hidden_ratio=fuse_hidden_ratio,
            fuse_init_open_bias=fuse_init_open_bias,
        )

        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(dims[0], num_classes, kernel_size=1, bias=True)

    def forward(
        self,
        x: torch.Tensor,  # NCHW
        return_aux: bool = False,
        gate_ref_pack: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        inp_hw = x.shape[-2:]

        if not return_aux:
            z = self.stem(x)
            x0 = self.enc0(z)
            x1 = self.enc1(self.down0(x0))
            x2 = self.enc2(self.down1(x1))
            x3 = self.enc3(self.down2(x2))

            y2 = self.dec2(x3, x2)
            y1 = self.dec1(y2, x1)
            y0 = self.dec0(y1, x0)

            y = self.up_final(y0)
            y = F.interpolate(y, size=inp_hw, mode="bilinear", align_corners=False)
            return self.seg_head(y)

        gate_ref_is_full = False
        if gate_ref_pack is not None:
            gate_ref_is_full = bool(gate_ref_pack.get("gate_ref_is_full", False))

        aux_total: Dict[str, Any] = {
            "moe_lb_loss": None,
            "gate_entropy": None,
            "expert_usage": None,
            "gate_probs_coarse": None,
            "gate_probs_full": None,
        }
        ood_cons_total: Optional[torch.Tensor] = None
        n_stages = 0

        def _accum(aux: Dict[str, Any]) -> None:
            nonlocal ood_cons_total, n_stages
            n_stages += 1

            lb = aux["moe_lb_loss"]
            aux_total["moe_lb_loss"] = lb if aux_total["moe_lb_loss"] is None else (aux_total["moe_lb_loss"] + lb)

            ge = aux.get("gate_entropy", None)
            if ge is not None:
                aux_total["gate_entropy"] = ge if aux_total["gate_entropy"] is None else (aux_total["gate_entropy"] + ge)

            eu = aux.get("expert_usage", None)
            if eu is not None:
                aux_total["expert_usage"] = eu if aux_total["expert_usage"] is None else (aux_total["expert_usage"] + eu)

            aux_total["gate_probs_coarse"] = aux.get("gate_probs_coarse", None)
            aux_total["gate_probs_full"] = aux.get("gate_probs_full", None)

            if "ood_consistency_loss" in aux:
                ood_cons_total = aux["ood_consistency_loss"] if ood_cons_total is None else (ood_cons_total + aux["ood_consistency_loss"])

        def _grp(key: str) -> Optional[List[Optional[torch.Tensor]]]:
            return None if gate_ref_pack is None else gate_ref_pack.get(key, None)

        z = self.stem(x)

        x0, a0 = self.enc0(z, return_aux=True, gate_ref_list=_grp("enc0"), gate_ref_is_full=gate_ref_is_full); _accum(a0)
        x1, a1 = self.enc1(self.down0(x0), return_aux=True, gate_ref_list=_grp("enc1"), gate_ref_is_full=gate_ref_is_full); _accum(a1)
        x2, a2 = self.enc2(self.down1(x1), return_aux=True, gate_ref_list=_grp("enc2"), gate_ref_is_full=gate_ref_is_full); _accum(a2)
        x3, a3 = self.enc3(self.down2(x2), return_aux=True, gate_ref_list=_grp("enc3"), gate_ref_is_full=gate_ref_is_full); _accum(a3)

        y2, d2 = self.dec2(x3, x2, return_aux=True, gate_ref_list=_grp("dec2"), gate_ref_is_full=gate_ref_is_full); _accum(d2)
        y1, d1 = self.dec1(y2, x1, return_aux=True, gate_ref_list=_grp("dec1"), gate_ref_is_full=gate_ref_is_full); _accum(d1)
        y0, d0 = self.dec0(y1, x0, return_aux=True, gate_ref_list=_grp("dec0"), gate_ref_is_full=gate_ref_is_full); _accum(d0)

        if aux_total["gate_entropy"] is not None and n_stages > 0:
            aux_total["gate_entropy"] = aux_total["gate_entropy"] / n_stages
        if aux_total["expert_usage"] is not None and n_stages > 0:
            aux_total["expert_usage"] = aux_total["expert_usage"] / n_stages

        if ood_cons_total is not None:
            aux_total["ood_consistency_loss"] = ood_cons_total

        y = self.up_final(y0)
        y = F.interpolate(y, size=inp_hw, mode="bilinear", align_corners=False)
        logits = self.seg_head(y)

        return logits, aux_total
