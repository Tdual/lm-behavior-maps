"""Local HuggingFace weights.

Only the logits at the continuation positions are materialised. Scoring every
position allocates `batch x sequence x vocab` floats -- 2.7 GB per call for a
152k-vocab model, and as much again for the log-softmax, which is enough to
push a laptop into swap. Slicing with `logits_to_keep` took one measured run
from over 54 minutes to 121 seconds, with a maximum difference of 9.5e-07.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFScorer:
    """Teacher-forced scoring with `transformers`.

    `chat` decides whether the model's chat template is applied. It defaults to
    "use it when the tokenizer has one", which is what a deployed agent does --
    and which is also why it belongs to the agent: applying or dropping the
    template moves a model about as far as swapping it for a different model.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "float32",
        chat: bool | None = None,
        prefix_space: bool = True,
        batch_size: int = 16,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = _pick_device(device)
        self.dtype_name = dtype
        self.prefix_space = prefix_space
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=getattr(torch, dtype),
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            .to(self.device)
            .eval()
        )
        has_template = getattr(self.tok, "chat_template", None) is not None
        self.chat = has_template if chat is None else (chat and has_template)
        self._torch = torch

    @property
    def fingerprint(self) -> str:
        return (
            f"hf:{self.model_id}:{self.dtype_name}"
            f":chat={int(self.chat)}:sp={int(self.prefix_space)}"
        )

    def close(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self.device == "mps":
            self._torch.mps.empty_cache()
        elif self.device == "cuda":
            self._torch.cuda.empty_cache()

    def _prefix_ids(self, body: str, system: str | None) -> list[int]:
        if self.chat:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": body}
            ]
            try:
                s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                return self.tok(s, add_special_tokens=False)["input_ids"]
            except Exception:
                pass
        pre = (system + "\n\n") if system else ""
        return self.tok(pre + body, add_special_tokens=True)["input_ids"]

    def loglikelihood(
        self, pairs: Sequence[tuple[str, str]], *, system: str | None = None
    ) -> list[float]:
        out: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            out.extend(self._batch(list(pairs[start : start + self.batch_size]), system))
        return out

    def _batch(self, pairs: list[tuple[str, str]], system: str | None) -> list[float]:
        torch = self._torch
        pre = [self._prefix_ids(b, system) for b, _ in pairs]
        cont = [
            self.tok((" " if self.prefix_space else "") + a, add_special_tokens=False)["input_ids"]
            for _, a in pairs
        ]
        seqs = [p + c for p, c in zip(pre, cont, strict=True)]
        length = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(seqs), length), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            att[i, : len(s)] = 1
        ids, att = ids.to(self.device), att.to(self.device)

        # The token at absolute position t is predicted by the logits at t-1,
        # so the kept window starts one before the shortest continuation.
        keep = length - min(len(p) for p in pre) + 1
        with torch.no_grad():
            try:
                logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=keep).logits
            except TypeError:
                logits = self.model(input_ids=ids, attention_mask=att).logits[:, -keep:, :]
            logp = torch.log_softmax(logits.float(), dim=-1)

        kept_from = length - keep
        out = np.empty(len(seqs), dtype=np.float64)
        for i, c in enumerate(cont):
            first = len(pre[i]) - 1 - kept_from  # row of the logits predicting c[0]
            tgt = torch.tensor(c, device=self.device)
            rows = logp[i, first : first + len(c), :]
            out[i] = rows.gather(1, tgt[:, None]).sum().item()
        return [float(v) for v in out]
