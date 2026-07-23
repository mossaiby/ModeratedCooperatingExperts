"""Model manager: loads/caches HuggingFace `transformers` models directly
(rather than via a server like Ollama/llama.cpp) so that later fine-tuning
(LoRA/PEFT) or access to model internals (hidden states, logits) remains
possible.

Models are configured per "role" (moderator, or a block type: text/code/
structured/image) via a YAML config file. Because a single consumer GPU
usually cannot hold every model simultaneously, an LRU cache with a
configurable `max_loaded_models` evicts the least-recently-used model when
the limit is exceeded.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_MAX_LOADED_MODELS = 2

# Third-party loggers/env vars that are extremely chatty during model
# download/load (HF Hub repo card fetches, tokenizer/config info logs,
# progress bars, tokenizer-parallelism fork warnings) but rarely useful
# unless actively debugging model loading itself.
_NOISY_LOGGERS = ("transformers", "huggingface_hub", "urllib3", "filelock")


def configure_model_logging(verbose: bool, debug: bool = False) -> None:
    """Silence (or re-enable) noisy transformers/huggingface_hub logging and
    progress bars. Call this once, before any model is loaded, based on the
    CLI's --verbose/--debug flags."""
    if debug:
        os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)
    elif verbose:
        os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)
    else:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)

    try:
        import transformers

        if debug:
            transformers.logging.set_verbosity_debug()
        elif verbose:
            transformers.logging.set_verbosity_info()
        else:
            transformers.logging.set_verbosity_error()
    except ImportError:
        pass  # transformers not installed yet / not needed for this call site


@dataclass
class ModelConfig:
    model_id: str
    device: str = "cpu"
    dtype: str = "auto"
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    kind: str = "causal_lm"
    """Either "causal_lm" (a chat-style transformers text model, the default)
    or "diffusion" (a diffusers text-to-image pipeline, used for the "image"
    role)."""


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


def load_model_configs(config_path: str | Path) -> dict[str, ModelConfig]:
    """Load a YAML file mapping role name (e.g. "moderator", "text", "code",
    "structured", "image") to a ModelConfig."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    max_loaded = raw.pop("max_loaded_models", None)  # tolerated, consumed by caller
    configs: dict[str, ModelConfig] = {}
    for role, cfg in raw.items():
        if role == "max_loaded_models":
            continue
        configs[role] = ModelConfig(
            model_id=cfg["model_id"],
            device=cfg.get("device", "cpu"),
            dtype=cfg.get("dtype", "auto"),
            generation_kwargs=cfg.get("generation_kwargs", {}) or {},
            kind=cfg.get("kind", "causal_lm"),
        )
    return configs


class ModelManager:
    """Lazily loads transformers models per role and evicts LRU entries when
    `max_loaded_models` is exceeded."""

    def __init__(
        self,
        configs: dict[str, ModelConfig],
        max_loaded_models: int = DEFAULT_MAX_LOADED_MODELS,
    ) -> None:
        self._configs = configs
        self._max_loaded_models = max_loaded_models
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ModelManager":
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        max_loaded = raw.get("max_loaded_models", DEFAULT_MAX_LOADED_MODELS)
        return cls(load_model_configs(config_path), max_loaded_models=max_loaded)

    def _load(self, role: str) -> LoadedModel:
        if role not in self._configs:
            raise KeyError(f"no model configured for role '{role}'")
        cfg = self._configs[role]

        if cfg.kind == "diffusion":
            return self._load_diffusion(cfg)
        return self._load_causal_lm(cfg)

    def _load_causal_lm(self, cfg: ModelConfig) -> LoadedModel:
        # Imported lazily so importing moce doesn't require torch/transformers
        # unless a real model is actually loaded (keeps unit tests fast/mockable).
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading causal LM for role: %s", cfg.model_id)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        dtype = None if cfg.dtype == "auto" else getattr(torch, cfg.dtype)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype)
        model.to(cfg.device)
        return LoadedModel(model=model, tokenizer=tokenizer)

    def _load_diffusion(self, cfg: ModelConfig) -> LoadedModel:
        # Imported lazily for the same reason as above; diffusers is only
        # required if an "image" (or other diffusion-kind) role is used.
        import torch
        from diffusers import AutoPipelineForText2Image

        logger.info("Loading text-to-image pipeline: %s", cfg.model_id)
        dtype = None if cfg.dtype == "auto" else getattr(torch, cfg.dtype)
        pipe = AutoPipelineForText2Image.from_pretrained(cfg.model_id, torch_dtype=dtype)
        pipe.to(cfg.device)
        return LoadedModel(model=pipe, tokenizer=None)

    def get(self, role: str) -> LoadedModel:
        if role in self._cache:
            self._cache.move_to_end(role)
            return self._cache[role]

        loaded = self._load(role)
        self._cache[role] = loaded
        self._cache.move_to_end(role)

        while len(self._cache) > self._max_loaded_models:
            evicted_role, _ = self._cache.popitem(last=False)
            logger.info("Evicting model for role '%s' (LRU cache full)", evicted_role)

        return loaded

    def generate(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        **generation_kwargs: Any,
    ) -> str:
        """Generate text for the given role using a chat-formatted prompt."""
        loaded = self.get(role)
        cfg = self._configs[role]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = loaded.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(cfg.device)

        merged_kwargs = {**cfg.generation_kwargs, **generation_kwargs}
        output_ids = loaded.model.generate(**inputs, **merged_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_length:]
        return loaded.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_image(
        self,
        role: str,
        prompt: str,
        output_path: str | Path,
        **generation_kwargs: Any,
    ) -> str:
        """Generate an image for `prompt` using the diffusion pipeline
        configured for `role`, saving it to `output_path` and returning the
        path (as a string)."""
        loaded = self.get(role)
        cfg = self._configs[role]
        merged_kwargs = {**cfg.generation_kwargs, **generation_kwargs}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = loaded.model(prompt, **merged_kwargs)
        image = result.images[0]
        image.save(output_path)
        return str(output_path)

