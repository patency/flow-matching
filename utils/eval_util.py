# metrics_utils.py
import os
import glob
from typing import List, Tuple, Optional
import math
from numbers import Number
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import torch

# -----------------------------
# Global cache/download location
# -----------------------------
DEFAULT_MODEL_DIR = os.path.abspath("./models")
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

# Put all caches here (FID / torch hub / misc libs)
os.environ["TORCH_HOME"] = DEFAULT_MODEL_DIR
os.environ["XDG_CACHE_HOME"] = DEFAULT_MODEL_DIR
# optional, harmless if unused:
os.environ["HF_HOME"] = os.path.join(DEFAULT_MODEL_DIR, "hf")

import torch
from torchvision.io import read_image

# torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")



# -----------------------------
# IO helpers
# -----------------------------
def list_image_paths(folder: str, exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "webp")) -> List[str]:
    """List image file paths in a folder (sorted)."""
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*.{e}")))
        paths.extend(glob.glob(os.path.join(folder, f"*.{e.upper()}")))
    paths = sorted(paths)
    if len(paths) == 0:
        raise RuntimeError(f"No images found in: {folder}")
    return paths


def load_image_uint8(path: str, device: torch.device) -> torch.Tensor:
    """Load an image as uint8 tensor (C,H,W) on device."""
    img = read_image(path).to(device)  # uint8, 0..255, (C,H,W)
    return img


def to_float01(img_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (C,H,W) -> float (1,C,H,W) in [0,1]."""
    return (img_uint8.float() / 255.0).unsqueeze(0)


def ensure_rgb_uint8(img_uint8: torch.Tensor) -> torch.Tensor:
    """Ensure image is 3-channel uint8 (C,H,W)."""
    if img_uint8.shape[0] == 1:
        img_uint8 = img_uint8.repeat(3, 1, 1)
    elif img_uint8.shape[0] > 3:
        img_uint8 = img_uint8[:3]  # safety: keep first 3 channels
    return img_uint8


def assert_pair_count(gt_paths: List[str], pred_paths: List[str]) -> None:
    if len(gt_paths) != len(pred_paths):
        raise RuntimeError(f"Count mismatch: gt={len(gt_paths)} vs pred={len(pred_paths)}")


# -----------------------------
# Metric primitives (single pair)
# -----------------------------
@torch.no_grad()
def psnr_pair(gt_uint8: torch.Tensor, pred_uint8: torch.Tensor, data_range: float = 1.0) -> float:
    """
    PSNR for one pair.
    Input: uint8 (C,H,W) tensors in [0,255].
    """
    gt = to_float01(gt_uint8)
    pr = to_float01(pred_uint8)
    return float(peak_signal_noise_ratio(pr, gt, data_range=data_range).item())


@torch.no_grad()
def ssim_pair(gt_uint8: torch.Tensor, pred_uint8: torch.Tensor, data_range: float = 1.0) -> float:
    """
    SSIM for one pair.
    Input: uint8 (C,H,W) tensors in [0,255].
    """
    gt = to_float01(gt_uint8)
    pr = to_float01(pred_uint8)
    return float(structural_similarity_index_measure(pr, gt, data_range=data_range).item())


# -----------------------------
# Metric over folders (mean)
# -----------------------------
@torch.no_grad()
def compute_psnr_folder(
    gt_dir: str,
    pred_dir: str,
    device: torch.device,
    exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "webp"),
) -> float:
    """Mean PSNR over paired images in two folders (sorted pairing)."""
    gt_paths = list_image_paths(gt_dir, exts)
    pred_paths = list_image_paths(pred_dir, exts)
    assert_pair_count(gt_paths, pred_paths)

    total = 0.0
    for g, p in zip(gt_paths, pred_paths):
        gt = load_image_uint8(g, device)
        pr = load_image_uint8(p, device)
        total += psnr_pair(gt, pr)
    return total / len(gt_paths)


@torch.no_grad()
def compute_ssim_folder(
    gt_dir: str,
    pred_dir: str,
    device: torch.device,
    exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "webp"),
) -> float:
    """Mean SSIM over paired images in two folders (sorted pairing)."""
    gt_paths = list_image_paths(gt_dir, exts)
    pred_paths = list_image_paths(pred_dir, exts)
    assert_pair_count(gt_paths, pred_paths)

    total = 0.0
    for g, p in zip(gt_paths, pred_paths):
        gt = load_image_uint8(g, device)
        pr = load_image_uint8(p, device)
        total += ssim_pair(gt, pr)
    return total / len(gt_paths)


@torch.no_grad()
def compute_fid_folder(
    gt_dir: str,
    pred_dir: str,
    device: torch.device,
    exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "webp"),
    feature: int = 2048,
) -> float:
    """
    FID between two folders.
    Uses torchmetrics' FrechetInceptionDistance.
    Expects uint8 [0,255]. Will auto-convert grayscale to RGB.
    """
    gt_paths = list_image_paths(gt_dir, exts)
    pred_paths = list_image_paths(pred_dir, exts)

    fid = FrechetInceptionDistance(feature=feature).to(device)

    for p in gt_paths:
        img = ensure_rgb_uint8(load_image_uint8(p, device))
        fid.update(img.unsqueeze(0), real=True)

    for p in pred_paths:
        img = ensure_rgb_uint8(load_image_uint8(p, device))
        fid.update(img.unsqueeze(0), real=False)

    return float(fid.compute().item())


# -----------------------------
# Convenience wrapper (optional)
# -----------------------------
@torch.no_grad()
def compute_all_metrics_folder(
    gt_dir: str,
    pred_dir: str,
    device: torch.device,
    exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "webp"),
) -> dict:
    """
    Convenience wrapper. (optional)
    Returns: {"psnr":..., "ssim":..., "fid":...}
    """
    return {
        "psnr": compute_psnr_folder(gt_dir, pred_dir, device, exts),
        "ssim": compute_ssim_folder(gt_dir, pred_dir, device, exts),
        "fid": compute_fid_folder(gt_dir, pred_dir, device, exts),
    }


def print_stats(
    x: Any,
    name: str = "value",
    expect_range: Tuple[float, float] = (-1.0, 1.0),
    per_channel: bool = True,
    quantiles=(0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0),
):
    """
    Print statistics for:
      - torch.Tensor (B,C,H,W) / (C,H,W) / any shape
      - scalar numbers: int/float/np.number
      - list/tuple/numpy array (will be treated as a flat array)

    Args:
        x: tensor / scalar / list / numpy array
        name: name to print
        expect_range: expected value range
        per_channel: tensor-only; whether to print per-channel stats when plausible
        quantiles: quantiles to report (tensor/array)
    """

    # ---- helper: scalar printing ----
    def _print_scalar(v: float):
        print(f"\n========== [{name}] ==========")
        print(f"type=scalar, value={v}")
        print(f"[global] min={v:.4f}, max={v:.4f}, mean={v:.4f}, std={0.0:.4f}")
        q_str = ", ".join([f"{q:.3f}:{v:.4f}" for q in quantiles])
        print(f"[quantile] {q_str}")
        lo, hi = expect_range
        out_low = 100.0 if v < lo else 0.0
        out_high = 100.0 if v > hi else 0.0
        print(f"[range check] <{lo}: {out_low:.2f}% | >{hi}: {out_high:.2f}%")
        print(f"[nan/inf] NaN={math.isnan(v)} Inf={math.isinf(v)}")
        print("=" * 32)

    # ---- case 1: python / numpy scalar ----
    if isinstance(x, Number) or isinstance(x, np.generic):
        _print_scalar(float(x))
        return

    # ---- case 2: convert list/tuple/numpy array to tensor-like ----
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.ndim == 0:  # scalar disguised
            _print_scalar(float(arr))
            return
        # treat as flat
        t_flat = torch.from_numpy(arr).float().reshape(-1)
        print(f"\n========== [{name}] ==========")
        print(f"type=array-like, shape={arr.shape}, dtype={arr.dtype}")
        minv = t_flat.min().item()
        maxv = t_flat.max().item()
        meanv = t_flat.mean().item()
        stdv = t_flat.std(unbiased=False).item()
        print(f"[global] min={minv:.4f}, max={maxv:.4f}, mean={meanv:.4f}, std={stdv:.4f}")
        qs = torch.tensor(quantiles)
        qv = torch.quantile(t_flat, qs).tolist()
        q_str = ", ".join([f"{q:.3f}:{v:.4f}" for q, v in zip(quantiles, qv)])
        print(f"[quantile] {q_str}")
        lo, hi = expect_range
        out_low = (t_flat < lo).float().mean().item() * 100
        out_high = (t_flat > hi).float().mean().item() * 100
        print(f"[range check] <{lo}: {out_low:.2f}% | >{hi}: {out_high:.2f}%")
        print(f"[nan/inf] NaN={(~torch.isfinite(t_flat)).any().item() and torch.isnan(t_flat).any().item()} "
              f"Inf={(~torch.isfinite(t_flat)).any().item() and torch.isinf(t_flat).any().item()}")
        print("=" * 32)
        return

    # ---- case 3: tensor ----
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor / number / list / numpy array, got {type(x)}")

    with torch.no_grad():
        t = x.detach()
        print(f"\n========== [{name}] ==========")
        print(f"type=tensor, shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")

        t_flat = t.float().reshape(-1)
        minv = t_flat.min().item()
        maxv = t_flat.max().item()
        meanv = t_flat.mean().item()
        stdv = t_flat.std(unbiased=False).item()
        print(f"[global] min={minv:.4f}, max={maxv:.4f}, mean={meanv:.4f}, std={stdv:.4f}")

        qs = torch.tensor(quantiles, device=t_flat.device)
        qv = torch.quantile(t_flat, qs).cpu().tolist()
        q_str = ", ".join([f"{q:.3f}:{v:.4f}" for q, v in zip(quantiles, qv)])
        print(f"[quantile] {q_str}")

        lo, hi = expect_range
        out_low = (t_flat < lo).float().mean().item() * 100
        out_high = (t_flat > hi).float().mean().item() * 100
        print(f"[range check] <{lo}: {out_low:.2f}% | >{hi}: {out_high:.2f}%")

        print(f"[nan/inf] NaN={torch.isnan(t).any().item()} Inf={torch.isinf(t).any().item()}")

        # per-channel (only if it looks like (B,C,H,W) or (C,H,W))
        if per_channel and t.ndim >= 3:
            t_c = t.unsqueeze(0) if t.ndim == 3 else t
            C = t_c.shape[1]
            print("\n[per-channel]")
            for c in range(C):
                tc = t_c[:, c].float()
                print(
                    f"  C{c}: min={tc.min().item():.4f}, "
                    f"max={tc.max().item():.4f}, "
                    f"mean={tc.mean().item():.4f}, "
                    f"std={tc.std(unbiased=False).item():.4f}"
                )

        print("=" * 32)


import math
from numbers import Number
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import torch


def print_stats(
    x: Any,
    name: str = "value",
    expect_range: Tuple[float, float] = (-1.0, 1.0),
    per_channel: bool = True,
    quantiles=(0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0),
):
    """
    Print statistics for:
      - torch.Tensor (B,C,H,W) / (C,H,W) / any shape
      - scalar numbers: int/float/np.number
      - list/tuple/numpy array (will be treated as a flat array)

    Args:
        x: tensor / scalar / list / numpy array
        name: name to print
        expect_range: expected value range
        per_channel: tensor-only; whether to print per-channel stats when plausible
        quantiles: quantiles to report (tensor/array)
    """

    # ---- helper: scalar printing ----
    def _print_scalar(v: float):
        print(f"\n========== [{name}] ==========")
        print(f"type=scalar, value={v}")
        print(f"[global] min={v:.4f}, max={v:.4f}, mean={v:.4f}, std={0.0:.4f}")
        q_str = ", ".join([f"{q:.3f}:{v:.4f}" for q in quantiles])
        print(f"[quantile] {q_str}")
        lo, hi = expect_range
        out_low = 100.0 if v < lo else 0.0
        out_high = 100.0 if v > hi else 0.0
        print(f"[range check] <{lo}: {out_low:.2f}% | >{hi}: {out_high:.2f}%")
        print(f"[nan/inf] NaN={math.isnan(v)} Inf={math.isinf(v)}")
        print("=" * 32)

    # ---- case 1: python / numpy scalar ----
    if isinstance(x, Number) or isinstance(x, np.generic):
        _print_scalar(float(x))
        return

    # ---- case 2: convert list/tuple/numpy array to tensor-like ----
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.ndim == 0:  # scalar disguised
            _print_scalar(float(arr))
            return
        # treat as flat
        t_flat = torch.from_numpy(arr).float().reshape(-1)
        print(f"\n========== [{name}] ==========")
        print(f"type=array-like, shape={arr.shape}, dtype={arr.dtype}")
        minv = t_flat.min().item()
        maxv = t_flat.max().item()
        meanv = t_flat.mean().item()
        stdv = t_flat.std(unbiased=False).item()
        print(f"[global] min={minv:.4f}, max={maxv:.4f}, mean={meanv:.4f}, std={stdv:.4f}")
        qs = torch.tensor(quantiles)
        qv = torch.quantile(t_flat, qs).tolist()
        q_str = ", ".join([f"{q:.3f}:{v:.4f}" for q, v in zip(quantiles, qv)])
        print(f"[quantile] {q_str}")
        lo, hi = expect_range
        out_low = (t_flat < lo).float().mean().item() * 100
        out_high = (t_flat > hi).float().mean().item() * 100
        print(f"[range check] <{lo}: {out_low:.2f}% | >{hi}: {out_high:.2f}%")
        print(f"[nan/inf] NaN={(~torch.isfinite(t_flat)).any().item() and torch.isnan(t_flat).any().item()} "
              f"Inf={(~torch.isfinite(t_flat)).any().item() and torch.isinf(t_flat).any().item()}")
        print("=" * 32)
        return

    # ---- case 3: tensor ----
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor / number / list / numpy array, got {type(x)}")

    with torch.no_grad():
        t = x.detach()
        print(f"\n========== [{name}] ==========")
        print(f"type=tensor, shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")

        t_flat = t.float().reshape(-1)
        minv = t_flat.min().item()
        maxv = t_flat.max().item()
        meanv = t_flat.mean().item()
        stdv = t_flat.std(unbiased=False).item()
        print(f"[global] min={minv:.4f}, max={maxv:.4f}, mean={meanv:.4f}, std={stdv:.4f}")

        qs = torch.tensor(quantiles, device=t_flat.device)
        qv = torch.quantile(t_flat, qs).cpu().tolist()
        q_str = ", ".join([f"{q:.3f}:{v:.4f}" for q, v in zip(quantiles, qv)])
        print(f"[quantile] {q_str}")

        lo, hi = expect_range
        out_low = (t_flat < lo).float().mean().item() * 100
        out_high = (t_flat > hi).float().mean().item() * 100
        print(f"[range check] <{lo}: {out_low:.2f}% | >{hi}: {out_high:.2f}%")

        print(f"[nan/inf] NaN={torch.isnan(t).any().item()} Inf={torch.isinf(t).any().item()}")

        # per-channel (only if it looks like (B,C,H,W) or (C,H,W))
        if per_channel and t.ndim >= 3:
            t_c = t.unsqueeze(0) if t.ndim == 3 else t
            C = t_c.shape[1]
            print("\n[per-channel]")
            for c in range(C):
                tc = t_c[:, c].float()
                print(
                    f"  C{c}: min={tc.min().item():.4f}, "
                    f"max={tc.max().item():.4f}, "
                    f"mean={tc.mean().item():.4f}, "
                    f"std={tc.std(unbiased=False).item():.4f}"
                )

        print("=" * 32)
