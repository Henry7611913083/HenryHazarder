"""
참고로 이거 클로드 소넷 4.5가 짰고
나머지 뒤는 Github 코파일럿에 있는 클로드 하이쿠 3.5가 짜고
다시 클로드 소넷 4.5가 짜고
제가 확인했어요, 누군가 고쳐주신다면 감사하겠습니다

nsfw_attack.py
--------------
Gradient-based adversarial perturbation that maximizes the NSFW class score
of Falconsai/nsfw_image_detection while minimizing perceptual distortion.

Method: PGD (Projected Gradient Descent) + LPIPS perceptual regularization
  L_total = -P(nsfw) + lambda_lpips * L_LPIPS + mu_l2 * L_L2
  Perturbation is projected onto the Linf ball [-eps, +eps] after each step.

Robustness enhancements:
  - Multi-scale evaluation (--multi-scale)
  - JPEG compression robustness (--use-jpeg)
  - Geometric transformations (--use-transforms)
  - DCT high-frequency preservation (--preserve-hf)
  - High-quality resampling (Lanczos)

Requirements:
  uv pip install torch torchvision transformers pillow lpips tqdm scipy numpy
  (CUDA is used automatically when available)

Usage:
  python nsfw_attack.py [--eps 0.03] [--steps 100] [--lr 0.005]
                        [--lambda-lpips 2.0] [--mu-l2 0.5]
                        [--multi-scale] [--use-jpeg] [--use-transforms]
                        [--preserve-hf] [--input ./input] [--output ./output]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Optional dependencies
try:
    import lpips as _lpips_mod
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from scipy.fftpack import dct, idct
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nsfw_attack")

if not HAS_LPIPS:
    log.warning("lpips not found — falling back to L2 regularization only.")
    log.warning("  install with: uv pip install lpips")

if not HAS_SCIPY:
    log.warning("scipy not found — DCT high-frequency preservation disabled.")
    log.warning("  install with: uv pip install scipy")


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save CHW float [0, 1] tensor as PNG."""
    arr = tensor.detach().cpu().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def tensor_from_pil(img: Image.Image) -> torch.Tensor:
    """PIL image → CHW float [0, 1] tensor."""
    return transforms.ToTensor()(img)


# ──────────────────────────────────────────────
# High-Quality Resampling
# ──────────────────────────────────────────────

def high_quality_resize(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Lanczos-based high-quality resampling to minimize information loss.
    tensor: CHW [0, 1]
    """
    pil_img = transforms.ToPILImage()(tensor.cpu())
    resized_pil = pil_img.resize(target_size, Image.LANCZOS)
    return transforms.ToTensor()(resized_pil).to(tensor.device)


def adaptive_downsample(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    num_steps: int = 3,
) -> torch.Tensor:
    """
    Gradual downsampling via multiple Lanczos steps to reduce cumulative error.
    tensor: CHW [0, 1]
    """
    current = tensor.clone()
    h_start, w_start = tensor.shape[1], tensor.shape[2]
    h_target, w_target = target_size

    for step in range(1, num_steps + 1):
        # Linear interpolation of target size
        progress = step / num_steps
        h_inter = int(h_start + (h_target - h_start) * progress)
        w_inter = int(w_start + (w_target - w_start) * progress)

        pil_img = transforms.ToPILImage()(current.cpu())
        current = transforms.ToTensor()(
            pil_img.resize((w_inter, h_inter), Image.LANCZOS)
        ).to(tensor.device)

    return current


# ──────────────────────────────────────────────
# DCT High-Frequency Preservation
# ──────────────────────────────────────────────

def preserve_high_frequency_dct(
    delta: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Amplify high-frequency components in DCT domain before resampling.
    This helps the perturbation survive downsampling/upsampling.

    delta: CHW tensor
    alpha: amplification factor (0.5 = 50% boost to high-freq)
    """
    if not HAS_SCIPY:
        log.warning("scipy unavailable; skipping DCT high-frequency preservation")
        return delta

    delta_np = delta.cpu().detach().numpy()
    delta_preserved = np.zeros_like(delta_np)

    for c in range(delta_np.shape[0]):
        # 2D DCT
        delta_freq = dct(dct(delta_np[c], axis=0, norm="ortho"), axis=1, norm="ortho")

        # Amplify high-frequency (upper-right quadrant)
        h, w = delta_freq.shape
        delta_freq[h // 4 :, w // 4 :] *= (1 + alpha)

        # Inverse DCT
        delta_preserved[c] = idct(
            idct(delta_freq, axis=0, norm="ortho"), axis=1, norm="ortho"
        )

    return torch.from_numpy(delta_preserved).float().to(delta.device)


# ──────────────────────────────────────────────
# JPEG Compression Robustness
# ──────────────────────────────────────────────

def apply_jpeg_compression(
    tensor: torch.Tensor,
    quality: int = 85,
) -> torch.Tensor:
    """
    Simulate JPEG compression to test robustness.
    tensor: CHW [0, 1]
    """
    import io

    pil_img = transforms.ToPILImage()(tensor.cpu())
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return transforms.ToTensor()(Image.open(buffer).convert("RGB")).to(tensor.device)


# ──────────────────────────────────────────────
# Geometric Transformations
# ──────────────────────────────────────────────

def apply_geometric_transform(
    tensor: torch.Tensor,
    angle: float = 0,
    scale: float = 1.0,
    translate: tuple[float, float] = (0, 0),
) -> torch.Tensor:
    """
    Apply affine transformation: rotation, scaling, translation.
    tensor: CHW [0, 1]
    """
    return transforms.functional.affine(
        tensor,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=0,
        resample=Image.LANCZOS,
    )


# ──────────────────────────────────────────────
# Score Computation (with optional transformations)
# ──────────────────────────────────────────────

def compute_scores(
    model,
    feature_extractor,
    tensor: torch.Tensor,
    id2label: dict,
    device: torch.device,
) -> dict[str, float]:
    """Compute NSFW scores for a tensor."""
    pil = transforms.ToPILImage()(tensor.cpu())
    inputs = feature_extractor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = F.softmax(logits, dim=-1)[0]
    return {id2label[i]: probs[i].item() for i in range(len(probs))}


def compute_scores_single_class(
    model,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    input_size: int,
    tensor: torch.Tensor,
    nsfw_class_idx: int,
    device: torch.device,
) -> float:
    """Compute NSFW score — PGD 루프와 동일한 전처리 경로 사용."""
    x = tensor.to(device)
    with torch.no_grad():
        x_resized = F.interpolate(
            x.unsqueeze(0),
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        pixel_values = (x_resized - mean_t) / std_t
        logits = model(pixel_values=pixel_values).logits
        probs = F.softmax(logits, dim=-1)[0]
    return probs[nsfw_class_idx].item()


# ──────────────────────────────────────────────
# PGD Attack (Enhanced)
# ──────────────────────────────────────────────

def pgd_attack(
    ensemble: list[tuple],
    nsfw_class_indices: list[int],
    input_sizes: list[int],
    orig_tensor: torch.Tensor,
    device: torch.device,
    image_name: str,
    eps: float = 0.03,
    steps: int = 100,
    lr: float = 0.005,
    lambda_lpips: float = 2.0,
    mu_l2: float = 0.5,
    lpips_net=None,
    accelerator=None,  # ✓ 추가
) -> torch.Tensor:
    """PGD attack with optional accelerator support."""
    orig = orig_tensor.to(device)
    delta = torch.zeros_like(orig, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([delta], lr=lr)
    step_w = len(str(steps))

    norm_params = [
        (
            torch.tensor(fe.image_mean).view(1, 3, 1, 1).to(device),
            torch.tensor(fe.image_std).view(1, 3, 1, 1).to(device),
        )
        for _, fe, m in ensemble
    ]
    for _, _, m in ensemble:
        m.eval()

    # ✓ accelerator.print() 사용 (main process만 출력)
    log_fn = accelerator.print if accelerator else log.info
    log_fn(f"[{image_name}] starting PGD  steps={steps}  eps={eps}  lr={lr}")

    t0 = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        adv_native = orig + delta

        nsfw_probs: list[torch.Tensor] = []
        for (_, _, m), nsfw_idx, (mean_t, std_t), sz in zip(
            ensemble, nsfw_class_indices, norm_params, input_sizes
        ):
            adv_resized = F.interpolate(
                adv_native.unsqueeze(0),
                size=(sz, sz),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            pixel_values = (adv_resized.unsqueeze(0) - mean_t) / std_t
            logits = m(pixel_values=pixel_values).logits
            prob = F.softmax(logits, dim=-1)[0][nsfw_idx]
            nsfw_probs.append(prob)

        nsfw_prob = torch.stack(nsfw_probs).mean()
        loss_cls = -nsfw_prob
        loss_l2 = (delta**2).mean()

        if lpips_net is not None:
            loss_lpips = lpips_net(orig.unsqueeze(0), adv_native.unsqueeze(0)).mean()
            loss = loss_cls + lambda_lpips * loss_lpips + mu_l2 * loss_l2
        else:
            loss_lpips = torch.tensor(0.0)
            loss = loss_cls + mu_l2 * loss_l2

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-eps, eps)

        nsfw_scores_str = "  ".join(f"{p.item():.4f}" for p in nsfw_probs)
        log_fn(
            f"[{image_name}] step [{step:0{step_w}d}/{steps}]  "
            f"loss={loss.item():.6f}  "
            f"loss_cls={loss_cls.item():.6f}  "
            f"loss_lpips={loss_lpips.item():.6f}  "
            f"loss_l2={loss_l2.item():.6f}  |  "
            f"nsfw: {nsfw_scores_str}"
        )

    elapsed = time.time() - t0
    log_fn(f"[{image_name}] PGD complete  elapsed={elapsed:.1f}s")

    with torch.no_grad():
        result = (orig + delta).cpu().clamp(0, 1)

    return result

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NSFW adversarial attack via PGD")
    parser.add_argument("--input", default="./input", help="Input image directory")
    parser.add_argument("--output", default="./output", help="Output image directory")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        metavar="MODEL_ID",
        help="HuggingFace model ID (여러 번 지정 가능 → 앙상블)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.03,
        help="Linf epsilon — max perturbation per pixel (default 0.03 ≈ 7.6/255)",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of PGD optimisation steps"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument(
        "--lambda-lpips",
        type=float,
        default=2.0,
        help="Weight for LPIPS perceptual loss",
    )
    parser.add_argument(
        "--mu-l2",
        type=float,
        default=0.5,
        help="Weight for L2 pixel regularisation",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Disable LPIPS (faster, less perceptual quality)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize images to square before processing (e.g. 224)",
    )

    # Robustness options
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable multi-scale robustness (test at 0.5x, 0.75x, 1.0x scales)",
    )
    parser.add_argument(
        "--use-jpeg",
        action="store_true",
        help="Enable JPEG compression robustness (qualities: 95, 85, 75)",
    )
    parser.add_argument(
        "--use-transforms",
        action="store_true",
        help="Enable geometric transformation robustness (rotation, scaling, translation)",
    )
    parser.add_argument(
        "--preserve-hf",
        action="store_true",
        help="Enable DCT high-frequency preservation (requires scipy)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Enable all robustness features (--multi-scale --use-jpeg --use-transforms --preserve-hf)",
    )

    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision="no",  # "fp16", "bf16", "fp8", "no" 중 선택
        gradient_accumulation_steps=1,
        cpu=False,
    )

    if args.models is None:
        args.models = ["Falconsai/nsfw_image_detection"]

    # Expand --robust flag
    if args.robust:
        args.multi_scale = True
        args.use_jpeg = True
        args.use_transforms = True
        args.preserve_hf = True

    run_start = time.time()

    device = accelerator.device  # ✓ Accelerator가 자동 결정

    accelerator.print(f"device: {device}")
    accelerator.print(f"distributed_type: {accelerator.distributed_type}")
    accelerator.print(f"num_processes: {accelerator.num_processes}")

    # ── Paths ──
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        log.error(f"no images found in '{input_dir}'")
        sys.exit(1)
    log.info(f"found {len(images)} image(s) in '{input_dir}'")

    # ── Ensemble 로딩 ──
    log.info(f"loading {len(args.models)} model(s)")
    ensemble: list[tuple] = []
    for model_id in args.models:
        fe = AutoImageProcessor.from_pretrained(model_id)
        m = AutoModelForImageClassification.from_pretrained(model_id)
        m = accelerator.prepare_model(m)  # ✓ 모델 준비
        m.eval()
        ensemble.append((model_id, fe, m))
        log.info(f"  loaded: {model_id}  labels={m.config.id2label}")

    if HAS_LPIPS and not args.no_lpips:
        log.info("loading LPIPS perceptual loss network")
        lpips_net = _lpips_mod.LPIPS(net="alex")
        lpips_net = accelerator.prepare_model(lpips_net)  # ✓ 모델 준비
        lpips_net.eval()

    norm_params = [
    (
        torch.tensor(fe.image_mean).view(1, 3, 1, 1).to(device),
        torch.tensor(fe.image_std).view(1, 3, 1, 1).to(device),
    )
    for _, fe, _ in ensemble
]

    input_sizes = [
        m.config.image_size if hasattr(m.config, "image_size") else 224
        for _, _, m in ensemble
    ]
    for (model_id, _, _), sz in zip(ensemble, input_sizes):
        accelerator.print(f"  [{model_id}] input size: {sz}×{sz}")

    # NSFW 클래스 인덱스 모델별로 resolve
    def resolve_nsfw_idx(m) -> int:
        for idx, label in m.config.id2label.items():
            if "nsfw" in label.lower():
                return int(idx)
        log.warning(f"no 'nsfw' label found in {m.config.id2label} — using index 1")
        return 1

    nsfw_indices = [resolve_nsfw_idx(m) for _, _, m in ensemble]
    for (model_id, _, m), nsfw_idx in zip(ensemble, nsfw_indices):
        log.info(f"  [{model_id}] NSFW class: index={nsfw_idx}  label='{m.config.id2label[nsfw_idx]}'")

    # ── LPIPS ──
    lpips_net = None
    if HAS_LPIPS and not args.no_lpips:
        log.info("loading LPIPS perceptual loss network")
        lpips_net = _lpips_mod.LPIPS(net="alex").to(device)
        lpips_net.eval()
        log.info(
            f"loss = -P(nsfw) + {args.lambda_lpips} * L_LPIPS + {args.mu_l2} * L_L2"
        )
    else:
        log.info(f"LPIPS disabled — loss = -P(nsfw) + {args.mu_l2} * L_L2")

    # ── Per-image loop ──
    results: list[dict] = []

    for img_idx, img_path in enumerate(images, start=1):
        log.info("─" * 72)
        log.info(f"processing image [{img_idx}/{len(images)}]: {img_path.name}")

        # ✓ 먼저 정의
        model_header = "  ".join(
            f"[{i}] {model_id}" for i, (model_id, _, _) in enumerate(ensemble)
        )
        accelerator.print(f"[{img_path.name}] models: {model_header}")
        log.info(f"[{img_path.name}] models: {model_header}")

        size = (args.resize, args.resize) if args.resize else None
        pil_img = load_image(img_path, size=size)
        orig_tensor = tensor_from_pil(pil_img)

        orig_nsfw_avg = sum(
            compute_scores_single_class(m, mean_t, std_t, sz, orig_tensor, nsfw_idx, device)
            for (_, _, m), (mean_t, std_t), sz, nsfw_idx
            in zip(ensemble, norm_params, input_sizes, nsfw_indices)
        ) / len(ensemble)

        adv_tensor = pgd_attack(
            ensemble=ensemble,
            nsfw_class_indices=nsfw_indices,
            input_sizes=input_sizes,
            orig_tensor=orig_tensor,
            device=device,
            image_name=img_path.name,
            eps=args.eps,
            steps=args.steps,
            lr=args.lr,
            lambda_lpips=args.lambda_lpips,
            mu_l2=args.mu_l2,
            lpips_net=lpips_net,
            accelerator=accelerator,  # ✓ 추가
        )

        # 공격 후 스코어 — 앙상블 평균
        adv_nsfw_avg = sum(
            compute_scores_single_class(m, mean_t, std_t, sz, adv_tensor, nsfw_idx, device)
            for (_, _, m), (mean_t, std_t), sz, nsfw_idx
            in zip(ensemble, norm_params, input_sizes, nsfw_indices)
        ) / len(ensemble)

        delta_nsfw = adv_nsfw_avg - orig_nsfw_avg

        log.info(
            f"[{img_path.name}] adversarial nsfw (ensemble avg): {adv_nsfw_avg:.4f}"
        )
        log.info(
            f"[{img_path.name}] NSFW delta: "
            f"{orig_nsfw_avg:.4f} → {adv_nsfw_avg:.4f} ({delta_nsfw:+.4f})"
        )

        if accelerator.is_main_process:
            out_path = output_dir / (img_path.stem + "_adv.png")
            save_image(adv_tensor, out_path)
            accelerator.print(f"[{img_path.name}] saved → {out_path}")
        else:
            out_path = output_dir / (img_path.stem + "_adv.png")

        results.append(
            {
                "name": img_path.name,
                "orig_nsfw": orig_nsfw_avg,
                "adv_nsfw": adv_nsfw_avg,
                "delta_nsfw": delta_nsfw,
                "out_path": out_path,
            }
        )

    # ── Finished summary ──
    total_elapsed = time.time() - run_start
    avg_elapsed = total_elapsed / len(results)

    if accelerator.is_main_process:
        log.info("═" * 72)
        log.info("FINISHED")
        log.info(f"  total images  : {len(results)}")
        log.info(f"  total elapsed : {total_elapsed:.1f}s  ({avg_elapsed:.1f}s / image)")
        log.info(f"  output dir    : {output_dir.resolve()}")
        log.info("  results:")
    for r in results:
        log.info(
            f"    {r['name']:<32}  "
            f"nsfw {r['orig_nsfw']:.4f} → {r['adv_nsfw']:.4f}  "
            f"(Δ {r['delta_nsfw']:+.4f})  →  {r['out_path'].name}"
        )
    log.info("═" * 72)


if __name__ == "__main__":
    main()