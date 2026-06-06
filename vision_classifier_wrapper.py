"""
vision_classifier_wrapper.py
-----------------------------
Universal wrapper layer that normalises heterogeneous vision classifier
back-ends (HuggingFace AutoModelForImageClassification, timm-native, and
VLM zero-shot models such as CLIP / SigLIP / SigLIP2) into a single
interface consumed by pgd_attack() and score helpers.

Drop-in replacement for the (model_id, feature_extractor, model) tuple
ensemble used in classifier_attack.py.  All preprocessing, forward pass,
and probability extraction are encapsulated here so the PGD loop never
needs to know which back-end it is talking to.

Integration guide (classifier_attack.py diff summary)
------------------------------------------------------
  Line  54  — add imports:
              from vision_classifier_wrapper import (
                  VisionClassifierWrapper, build_wrapper, ActivationFn
              )

  Line  54  — keep existing imports; AutoImageProcessor /
              AutoModelForImageClassification are still used inside
              HFClassifierWrapper, so no removal needed.

  Lines 676-695  (ensemble loading loop) — replace with:
              wrappers: list[VisionClassifierWrapper] = [
                  build_wrapper(spec_str, accelerator)
                  for spec_str in args.specs   # raw "-S" strings
              ]

  Lines 713-733  (norm_params / input_sizes / target_labels) — delete;
              all three are now attributes on each wrapper instance.

  Line  752  (compute_scores_single_class call) — replace with:
              w.score(orig_tensor)

  Lines 764-781  (pgd_attack call) — change signature to:
              pgd_attack(wrappers=wrappers, orig_tensor=..., ...)

  Lines 344-496  (pgd_attack body) — replace inner loops with:
              x_in     = w.preprocess(adv_native)
              logits   = w.logits(x_in)
              loss_cls = w.cls_loss(logits, label_smooth)
              loss_kl  = w.kl_loss(logits, orig, kl_temp)

  Lines 784-788  (adv score) — replace with:
              w.score(adv_tensor)

Spec string format (extended, backward-compatible)
---------------------------------------------------
  hf:ORG/MODEL:TARGET          HuggingFace AutoModelForImageClassification
  timm:timm/MODEL_NAME:TARGET  timm model loaded via HF TimmWrapper
  vlm:ORG/MODEL:LABEL,LABEL    CLIP / SigLIP / SigLIP2 zero-shot
                                comma-separated candidate labels;
                                TARGET is the label to maximise

  Backward-compatible (no prefix) — treated as hf:
  ORG/MODEL:TARGET             same as hf:ORG/MODEL:TARGET

Requirements (additions to existing requirements)
-------------------------------------------------
  uv pip install timm  (only needed for timm: prefix)
"""

from __future__ import annotations

from abc import ABC, abstractmethod  # ← ABC, abstractmethod 정의 해결
from typing import Literal           # ← Literal 정의 해결

import logging
import torch
import torch.nn as nn                  # ← 추가: nn.Parameter 사용용
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from enum import Enum  # ← 상단 import 문에 추가

log = logging.getLogger("nsfw_attack")

class ActivationFn(str, Enum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    NONE = "none"


# ── Abstract base ──────────────────────────────────────────────────────────────

class VisionClassifierWrapper(ABC):
    """
    Common interface for all classifier back-ends.
    ...
    """

    model_id:   str
    target_idx: int
    input_size: int
    id2label:   dict[int, str]
    activation: ActivationFn

    # 🔽 [여기서부터 추가] 누락된 명시적 생성자 및 디바이스 제어 메서드 정의
    def __init__(
        self,
        model_id: str,
        target_idx: int,
        activation: ActivationFn = ActivationFn.SOFTMAX
    ):
        self.model_id = model_id
        self.target_idx = target_idx
        self.activation = activation
        
        # 하위 자식 클래스들이 파라미터 초기화 시 참조할 수 있도록 기본 디바이스 변수 선언
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "VisionClassifierWrapper":
        """Move any stored tensors (mean/std) and model to device. Returns self."""
        self._device = device
        # 내부에 _model 객체(HuggingFace 가중치 등)가 저장되어 있다면 해당 장치로 완전히 보냄
        if hasattr(self, "_model"):
            self._model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        """Expose the current device of the wrapper."""
        return self._device
    # 🔼 [여기까지 추가]

    # ── Interface methods (must be implemented) ────────────────────────────────

    @abstractmethod
    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a CHW float [0, 1] tensor (on any device) into the
        batched, normalised, resized input tensor expected by the model.

        Returns a [1, C, H, W] tensor on the model's device.

        IMPORTANT: implementation must use only differentiable torch ops
        (F.interpolate, arithmetic on tensors) so that gradients flow back
        through this function to the input tensor — which is required for
        the PGD optimisation loop.
        """

    @abstractmethod
    def logits(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        Run the model forward pass on a preprocessed [1, C, H, W] tensor.

        Returns raw (pre-activation) logits of shape [1, num_classes].
        For VLM zero-shot wrappers this is the cosine-similarity score
        vector (one entry per candidate label), still pre-activation.
        """

    # ── Derived helpers (shared across all back-ends) ──────────────────────────

    def probs(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        Apply the correct activation function to logits and return a
        probability vector of shape [num_classes].

        Uses sigmoid for SigLIP-family models (independent per-class
        probabilities) and softmax for all others (mutually exclusive).
        """
        raw = self.logits(preprocessed)  # [1, C]
        if self.activation == "sigmoid":
            return torch.sigmoid(raw)[0]
        return F.softmax(raw, dim=-1)[0]

    @torch.no_grad()
    def score(self, tensor: torch.Tensor) -> float:
        """
        Return the scalar target-class probability for a CHW [0, 1]
        tensor.  Runs under no_grad — used for before/after logging only,
        not inside the optimisation loop.
        """
        x = self.preprocess(tensor)
        return self.probs(x)[self.target_idx].item()

    def cls_loss(
        self,
        logits_adv: torch.Tensor,
        label_smooth: float = 0.1,
    ) -> torch.Tensor:
        """
        Soft-label classification loss that pushes probability mass onto
        target_idx while leaving a small epsilon on all other classes.

        For softmax models: KL( log_softmax(logits) || soft_target )
        For sigmoid models: binary cross-entropy on the target logit,
          which avoids the invalid assumption of mutual exclusivity.

        Parameters
        ----------
        logits_adv  : raw logits [1, C] from the adversarial image
        label_smooth: fraction of probability mass spread over non-target
                      classes (ignored for sigmoid mode)
        """
        if self.activation == "sigmoid":
            # Each class is treated as an independent binary problem.
            # We just maximise the target logit — equivalent to minimising
            # BCE with label = 1 for the target class.
            target_logit = logits_adv[0, self.target_idx].unsqueeze(0)
            target_label = torch.ones(1, device=logits_adv.device)
            return F.binary_cross_entropy_with_logits(target_logit, target_label)

        # Softmax branch — same formula as the original code
        num_classes = logits_adv.shape[1]
        soft_target = torch.full_like(
            logits_adv, label_smooth / max(num_classes - 1, 1)
        )
        soft_target[0, self.target_idx] = 1.0 - label_smooth
        log_probs = F.log_softmax(logits_adv, dim=1)
        return F.kl_div(log_probs, soft_target, reduction="batchmean")

    def kl_loss(
        self,
        logits_adv: torch.Tensor,
        orig_tensor: torch.Tensor,
        kl_temp: float = 2.0,
    ) -> torch.Tensor:
        """
        KL divergence between the original and adversarial output
        distributions, scaled by temperature kl_temp.

        Encourages the perturbed image to stay on the natural image
        manifold by penalising large distribution shifts.

        For sigmoid models the KL is computed per-class independently
        (Bernoulli KL) to stay consistent with the activation semantics.

        Parameters
        ----------
        logits_adv  : adversarial logits [1, C] — must carry grad
        orig_tensor : original CHW [0, 1] tensor (no grad needed)
        kl_temp     : temperature for softening both distributions
        """
        with torch.no_grad():
            x_orig      = self.preprocess(orig_tensor)
            logits_orig = self.logits(x_orig)

        if self.activation == "sigmoid":
            # Bernoulli KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))
            # numerically stable via torch built-ins
            p      = torch.sigmoid(logits_orig / kl_temp)        # [1, C]
            log_q  = F.logsigmoid(logits_adv  / kl_temp)         # [1, C]
            log_1q = F.logsigmoid(-logits_adv / kl_temp)         # [1, C]
            kl = p * (torch.log(p + 1e-8) - log_q) + (1 - p) * (
                torch.log(1 - p + 1e-8) - log_1q
            )
            return kl.mean()

        # Softmax branch — identical to original code
        p_orig    = F.softmax(logits_orig / kl_temp, dim=1)
        log_p_adv = F.log_softmax(logits_adv  / kl_temp, dim=1)
        return F.kl_div(log_p_adv, p_orig, reduction="batchmean")

    # ── Device placement ───────────────────────────────────────────────────────

    def to(self, device: torch.device) -> "VisionClassifierWrapper":
        """Move any stored tensors (mean/std) to device. Returns self."""
        return self  # subclasses override if they hold tensors


# ── Back-end 1: HuggingFace AutoModelForImageClassification ───────────────────

class HFClassifierWrapper(VisionClassifierWrapper):
    """
    Wrapper for HuggingFace AutoModelForImageClassification models.
    """
    def __init__(
        self,
        model_id: str,
        processor,
        model,
        target_idx: int,
        activation: ActivationFn = ActivationFn.SOFTMAX
    ):
        super().__init__(model_id, target_idx, activation)
        self._model = model
        self.id2label = model.config.id2label
        
        # 0. 디바이스 변수를 가중치 상태에 맞게 수동 초기화 (안전장치)
        self._device = next(model.parameters()).device

        # 1. 해상도(input_size) 파싱 초강력 방어 코드
        if hasattr(processor, "size") and isinstance(processor.size, dict):
            self.input_size = processor.size.get("height", 224)
        elif hasattr(processor, "size") and isinstance(processor.size, int):
            self.input_size = processor.size
        elif hasattr(model, "config") and hasattr(model.config, "image_size") and model.config.image_size:
            self.input_size = model.config.image_size
        elif hasattr(model, "timm_model") and hasattr(model.timm_model, "default_cfg"):
            # Timm 백엔드 래퍼 모델 구조인 경우 내부 default_cfg를 추적하여 448 스케일 강제 획득
            self.input_size = model.timm_model.default_cfg.get("input_size", (3, 224, 224))[-1]
        elif hasattr(model, "config") and hasattr(model.config, "hf_model_config") and hasattr(model.config.hf_model_config, "image_size"):
            self.input_size = model.config.hf_model_config.image_size
        else:
            # 모델 ID 이름에 '448' 문자열이 명시되어 있다면 휴리스틱하게 가로챈다
            if "448" in model_id:
                self.input_size = 448
            else:
                self.input_size = 224

        # 디버깅 및 명세 확인용 로그 강제 출력 (터미널에서 직접 확인용)
        import logging
        logging.info(f"  [{model_id}] Resolved execution input size: {self.input_size}x{self.input_size}")

        # 2. TimmWrapperImageProcessor 및 누락된 프로세서 속성 방어
        if hasattr(processor, "image_mean"):
            mean_val = processor.image_mean
            std_val = processor.image_std
        elif hasattr(processor, "image_processor") and hasattr(processor.image_processor, "mean"):
            mean_val = processor.image_processor.mean
            std_val = processor.image_processor.std
        else:
            # 최종 Fallback (표준 ImageNet/ViT 규격)
            mean_val = [0.5, 0.5, 0.5]
            std_val = [0.5, 0.5, 0.5]

        # 3. 모델 가중치가 배치된 디바이스 위치로 함께 주입
        self._mean = nn.Parameter(
            torch.tensor(mean_val, dtype=torch.float32).view(1, 3, 1, 1).to(self._device),
            requires_grad=False
        )
        self._std = nn.Parameter(
            torch.tensor(std_val, dtype=torch.float32).view(1, 3, 1, 1).to(self._device),
            requires_grad=False
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # 입력받은 텐서를 현재 모델이 동작하는 디바이스로 강제 이동
        x = x.to(self._device)
        
        if x.ndim == 3:
            x = x.unsqueeze(0)
            
        # 각 개별 모델의 목표 해상도(224 또는 448)에 맞춰 유연하게 리사이즈
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False
            )
            
        # 정규화 연산 (디바이스 일치 보장)
        normalized = (x - self._mean) / self._std
        return normalized

    def logits(self, preprocessed: torch.Tensor) -> torch.Tensor:
        # 전처리된 텐서 장치 상태 최종 확인 후 순방향 연산
        if preprocessed.device != self._device:
            preprocessed = preprocessed.to(self._device)
        return self._model(pixel_values=preprocessed).logits

    def to(self, device: torch.device) -> "HFClassifierWrapper":
        """장치 변경 명령이 내려올 때 가중치와 하이퍼파라미터를 동시 이동"""
        self._device = device
        self._model.to(device)
        self._mean = nn.Parameter(self._mean.to(device), requires_grad=False)
        self._std = nn.Parameter(self._std.to(device), requires_grad=False)
        return self


# ── Back-end 2: VLM zero-shot (CLIP / SigLIP / SigLIP2) ──────────────────────

class VLMZeroShotWrapper(VisionClassifierWrapper):
    """
    Zero-shot classifier backed by any dual-encoder VLM: CLIP, SigLIP,
    or SigLIP2.

    Classification is performed by computing cosine similarity between the
    image embedding and a set of pre-encoded text embeddings (one per
    candidate label), then treating the resulting similarity scores as
    logits.

    Activation function
    -------------------
    CLIP      → softmax  (contrastive, mutually exclusive)
    SigLIP/2  → sigmoid  (independent per-class probabilities)

    The correct activation is detected automatically by checking whether
    the model's config class name contains "siglip".

    Preprocessing
    -------------
    Image pixels are resized and normalised using the model processor's
    stored mean/std — same differentiable path as HFClassifierWrapper.
    Text embeddings are pre-computed once at construction time and cached.

    Parameters
    ----------
    model_id          : HuggingFace model identifier (for logging)
    model             : CLIPModel / SiglipModel / Siglip2Model instance
    processor         : matching AutoProcessor instance
    candidate_labels  : list of text strings, one per class
    target_label      : substring matched against candidate_labels to
                        identify which class index to maximise
    device            : torch.device for all tensor operations
    """

    def __init__(
        self,
        model_id: str,
        model,
        processor,
        candidate_labels: list[str],
        target_label: str,
        device: torch.device,
    ) -> None:
        self.model_id        = model_id
        self.id2label        = {i: lbl for i, lbl in enumerate(candidate_labels)}
        self._model          = model
        self._processor      = processor
        self._device         = device
        self.input_size      = self._resolve_input_size(processor)

        # Detect activation: SigLIP family uses sigmoid, all others softmax
        cfg_cls = type(model.config).__name__.lower()
        self.activation: ActivationFn = (
            "sigmoid" if "siglip" in cfg_cls else "softmax"
        )

        # Resolve target index from label substring match
        target_lower = target_label.lower()
        matches = [
            i for i, lbl in self.id2label.items()
            if target_lower in lbl.lower()
        ]
        if not matches:
            log.warning(
                f"[{model_id}] target '{target_label}' not found in "
                f"{list(self.id2label.values())} — falling back to index 0"
            )
            self.target_idx = 0
        else:
            self.target_idx = matches[0]

        # Pre-compute normalisation tensors
        img_mean = getattr(
            processor, "image_mean", [0.48145466, 0.4578275, 0.40821073]
        )
        img_std = getattr(
            processor, "image_std",  [0.26862954, 0.26130258, 0.27577711]
        )
        self._mean = (
            torch.tensor(img_mean, dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(device)
        )
        self._std = (
            torch.tensor(img_std, dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(device)
        )

        # Pre-encode all candidate labels once; shape [num_labels, embed_dim]
        # This is the only part that uses the text encoder — never repeated.
        self._text_embeds = self._encode_texts(candidate_labels)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_input_size(processor) -> int:
        """
        Extract the square input size from a VLM processor.
        Tries several attribute names used by different processor versions.
        """
        for attr in ("size", "image_size", "crop_size"):
            val = getattr(processor, attr, None)
            if val is None:
                continue
            if isinstance(val, int):
                return val
            if isinstance(val, dict):
                return val.get("shortest_edge", val.get("height", 224))
        return 224

    @torch.no_grad()
    def _encode_texts(self, labels: list[str]) -> torch.Tensor:
        """
        Tokenise and encode a list of label strings through the model's
        text encoder.  Returns L2-normalised embeddings [N, D].
        """
        # Construct natural-language prompts identical to zero-shot evaluation
        prompts = [f"a photo of {lbl}" for lbl in labels]
        inputs  = self._processor(
            text=prompts,
            padding="max_length",  # required by SigLIP tokeniser
            return_tensors="pt",
        ).to(self._device)
        # Both CLIPModel and SiglipModel expose get_text_features()
        embeds = self._model.get_text_features(**inputs)  # [N, D]
        return F.normalize(embeds, dim=-1)

    # ── VisionClassifierWrapper implementation ─────────────────────────────────

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalise a CHW [0, 1] tensor.
        Identical differentiable path to HFClassifierWrapper.preprocess().
        """
        x = tensor.to(self._device)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return (x - self._mean) / self._std

    def logits(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between the image embedding and all
        pre-cached text embeddings, scaled by the model's logit_scale.

        Returns [1, num_labels] — same shape contract as HF classifiers.
        """
        img_embed = self._model.get_image_features(pixel_values=preprocessed)
        img_embed = F.normalize(img_embed, dim=-1)  # [1, D]

        # Retrieve learned temperature; fall back to 1/0.07 ≈ 14.3 (CLIP default)
        logit_scale = getattr(self._model, "logit_scale", None)
        scale = (
            logit_scale.exp().clamp(max=100.0)
            if logit_scale is not None
            else torch.tensor(14.3, device=self._device)
        )

        # [1, D] @ [D, N] → [1, N]
        similarity = img_embed @ self._text_embeds.T
        return similarity * scale

    def to(self, device: torch.device) -> "VLMZeroShotWrapper":
        self._device      = device
        self._mean        = self._mean.to(device)
        self._std         = self._std.to(device)
        self._text_embeds = self._text_embeds.to(device)
        return self


# ── Factory ────────────────────────────────────────────────────────────────────

def _parse_extended_spec(spec: str) -> tuple[str, str, str, list[str]]:
    """
    Parse an extended spec string into (backend, model_id, target, extras).

    Format (colon-separated, left-to-right):
      [BACKEND:]MODEL_ID[:TARGET]

    BACKEND — optional prefix, one of: "hf", "timm", "vlm"
              omitted → treated as "hf" for backward compatibility
    MODEL_ID — HuggingFace repo identifier (may contain '/' and '.')
    TARGET   — class label substring, integer index, or comma-separated
                label list (vlm only)

    Examples
    --------
      "Falconsai/nsfw_image_detection:nsfw"
          → ("hf",   "Falconsai/nsfw_image_detection", "nsfw",  [])
      "hf:AdamCodd/vit-base-nsfw-detector:nsfw"
          → ("hf",   "AdamCodd/vit-base-nsfw-detector",  "nsfw",  [])
      "timm:timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k:283"
          → ("timm", "timm/convnext_xxlarge.*",           "283",   [])
      "vlm:openai/clip-vit-base-patch32:nsfw,explicit,safe"
          → ("vlm",  "openai/clip-vit-base-patch32",     "nsfw",
             ["nsfw", "explicit", "safe"])
    """
    KNOWN_BACKENDS = {"hf", "timm", "vlm"}

    # Check whether the first colon-delimited token is a backend prefix
    parts = spec.split(":", 1)
    if len(parts) == 2 and parts[0].lower() in KNOWN_BACKENDS:
        backend   = parts[0].lower()
        remainder = parts[1]
    else:
        backend   = "hf"
        remainder = spec

    # The remainder may be  MODEL_ID  or  MODEL_ID:TARGET
    # Use rfind so HF org/repo paths with '/' are safe against mis-splitting
    colon = remainder.rfind(":")
    if colon == -1:
        model_id = remainder
        target   = ""
    else:
        model_id = remainder[:colon]
        target   = remainder[colon + 1:]

    # For vlm backend the target field is comma-separated candidate labels
    extras = [lbl.strip() for lbl in target.split(",")] if backend == "vlm" else []

    return backend, model_id, target, extras

def parse_spec(spec_str: str) -> dict:
    """
    Parses a spec string like 'Falconsai/nsfw_image_detection:nsfw' or 
    'backend@model_id:target_cls:activation' into a standardized spec dict.
    """
    # 기본값 설정
    backend = "hf"
    activation = ActivationFn.SOFTMAX  # 혹은 정의된 Enum 기본값
    
    # 래퍼 규격 파싱 로직 구현부
    if "@" in spec_str:
        backend, spec_str = spec_str.split("@", 1)
        
    target_cls = None
    if ":" in spec_str:
        model_id, target_cls = spec_str.split(":", 1)
    else:
        model_id = spec_str
        
    return {
        "backend": backend,
        "model_id": model_id,
        "target_cls": target_cls,
        "activation": activation
    }

def build_wrapper(spec_str: str, accelerator=None) -> VisionClassifierWrapper:
    """
    Parses a spec string and constructs the appropriate VisionClassifierWrapper.
    """
    spec = parse_spec(spec_str)
    backend = spec["backend"]
    model_id = spec["model_id"]
    target_cls = spec["target_cls"]
    activation = spec["activation"]

    device = accelerator.device if accelerator is not None else torch.device("cpu")

    if backend == "hf":
        # 모델을 먼저 로드합니다.
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
        model.eval()

        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            if processor is None:
                raise ValueError("Processor returned None")
        except Exception:
            # Freepik 모델처럼 preprocessor_config.json이 유실된 경우의 핵심 오버라이드
            logging.warning(f"[{model_id}] preprocessor_config.json 누락으로 표준 프로세서 구성을 재 매핑합니다.")
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            
            # 모델 명세(EVA-ViT 448) 정보를 프로세서 크기에 강제 동기화
            if hasattr(model.config, "image_size"):
                target_sz = model.config.image_size
                if isinstance(target_sz, int):
                    processor.size = {"height": target_sz, "width": target_sz}

        target_idx = _resolve_target_idx_from_model(model, target_cls)
        
        # 래퍼 인스턴스화
        wrapper = HFClassifierWrapper(
            model_id=model_id,
            processor=processor,
            model=model,
            target_idx=target_idx,
            activation=activation
        )
        
        # 생성된 래퍼 내부 파라미터(mean, std 등)들을 가속기 디바이스와 완벽 동기화
        wrapper.to(device)

        logging.info(
            f"  loaded: {model_id}  target_idx={wrapper.target_idx}  "
            f"target_label='{wrapper.id2label[wrapper.target_idx]}'  "
            f"all_labels={wrapper.id2label}"
        )
        return wrapper

def _resolve_target_idx_from_model(model, target: str | int | None) -> int:
    """
    Resolve a class index from a loaded HF model.

    Extracted from the original resolve_target_idx() so it can be called
    inside build_wrapper() without requiring a separate function call.

    Rules (unchanged from original classifier_attack.py)
    -----------------------------------------------------
    int  → validated against id2label, returned as-is
    str  → case-insensitive substring search in id2label values
    None → falls back to searching for "nsfw"
    """
    id2label: dict = model.config.id2label

    if isinstance(target, int):
        if target not in id2label:
            raise ValueError(
                f"Target index {target} out of range for id2label: {id2label}"
            )
        return target

    # Accept numeric strings passed from the spec parser (e.g. ":1")
    if isinstance(target, str) and target.isdigit():
        idx = int(target)
        if idx not in id2label:
            raise ValueError(
                f"Target index {idx} out of range for id2label: {id2label}"
            )
        return idx

    search = (target or "nsfw").lower()
    for idx, label in id2label.items():
        if search in label.lower():
            return int(idx)

    log.warning(
        f"Label '{search}' not found in {id2label} — falling back to index 1"
    )
    return 1