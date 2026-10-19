"""Optional OpenAI-compatible client with a safe local fallback."""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


class LLMClient:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def complete(self, system: str, user: str) -> str:
        if not self.enabled:
            return "Use the forward model after checking KG and RAG evidence."
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}/chat/completions", data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode())
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LLM response does not contain choices[0].message.content") from exc


class QwenLocalClient:
    """Lazy local Qwen inference through Transformers.

    The first call downloads the configured model into ``cache_dir``.  Lazy
    loading keeps CLI help, data checks and offline tests fast.
    """
    def __init__(self, model_name: str, cache_dir: str | None = None, device: str = "auto",
                 max_new_tokens: int = 768, temperature: float = 0.0, auto_install: bool = True,
                 dtype: str = "float16"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.auto_install = auto_install
        self.dtype = dtype
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            if not self.auto_install:
                raise RuntimeError("transformers is missing; run python -m LLM_agent.bootstrap")
            from .bootstrap import install_requirements
            from pathlib import Path
            install_requirements(Path(__file__).with_name("requirements.txt"))
            from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        model_kwargs = dict(kwargs)
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        import torch
        if self.dtype == "float16" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bfloat16" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device != "auto":
            self._model.to(self.device)
        self._model.eval()

    def complete(self, system: str, user: str) -> str:
        if self._model is None:
            self._load()
        import torch
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        target_device = next(self._model.parameters()).device
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                          do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5))
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
