#!/usr/bin/env python3
# ============================================================================#
#
#  Buy the Rumor, Sell the News: When Is News Priced In?
#  --  distilroberta_student.py
#
#  The study's student as a real network: distilroberta-base with masked
#  mean pooling and seven linear heads, implementing exactly the
#  specification Task 14 audits, behind the four student services of the
#  usage example (Step 3.2):
#  • ``student_network``    the torch.nn.Module Tasks 15 and 17 train
#  • ``student_model``      Task 16's first-pass scorer (texts -> arrays)
#  • ``student_predict_fn`` Tasks 17 and 18's scorer (frame -> head columns)
#  • ``student_encode``     Task 17's tokeniser
#  plus the ``encode`` / ``decode`` pair Task 15's training config carries.
#
#  The specification it implements (Task 14, DEF8, DEF17, DEF18, DEF20):
#  • Backbone: 6 layers, hidden 768, 12 heads of width 64, FFN 3,072, 514
#    positions, byte-level BPE vocabulary of 50,265; no pooler.
#  • Pooling: the masked mean of the final hidden states (DEF17).
#  • Heads: primary_tag 768 -> 17 (softmax), secondary_tag 768 -> 18
#    (softmax, 17 tags + "none", DEF8), and one 768 -> 1 sigmoid head per
#    attribute, named exactly as Task 14's parameter inventory names them.
#  • Output: forward(input_ids, attention_mask) returns (batch, 40) logits
#    ordered primary (17), secondary (18), attributes (5), the layout Task 15
#    slices for its DEF20 joint loss.
#  • Inference input: Task 15's numeral-preserving truncation (DEF18), so
#    the scorers see the same text the network was trained on.
#
#  Three conventions of the notebook that a real RoBERTa must absorb:
#  • Task 15 encodes text pieces separately and concatenates them, so
#    ``encode`` adds no special tokens; the network adds <s> and </s> itself.
#  • Task 15 sends up to 512 content tokens, but RoBERTa's 514-row position
#    table holds only 512 positions once its padding offset of 2 is taken,
#    including <s> and </s>. The network keeps the first 510 content tokens.
#  • Task 15 defaults and Task 17 hard-codes pad_token_id = 0, which is
#    RoBERTa's <s>, not its pad (1). The network therefore rebuilds every
#    sequence from ``attention_mask`` and never trusts the pad id, and it
#    passes explicit position ids so padding cannot shift them.
#
#  One known deviation from Task 14's inventory: the real distilroberta-base
#  checkpoint carries a 1 x 768 token-type table (type_vocab_size = 1)
#  where Task 14 pins type_vocab_size = 0. It adds 768 parameters
#  (81,558,568 against the inventory's 81,557,800, both inside Task 14's
#  +/- 1,000,000 band around the paper's 81,496,320) and is reported, not
#  hidden, by :func:`check_against_task14_inventory`.
#
#  Author: CS Chirinda
#  License: MIT
#
# ============================================================================#

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import RobertaConfig, RobertaModel
except ImportError as _error:  # pragma: no cover - exercised only without the deps
    raise ImportError(
        "distilroberta_student requires torch and transformers: "
        "python -m pip install torch transformers"
    ) from _error


# ----------------------------------------------------------------------------
# The frozen student contract (mirrors Tasks 14 and 15)
# ----------------------------------------------------------------------------

#: The Hugging Face identifier of the backbone.
BACKBONE_NAME: str = "distilbert/distilroberta-base"

#: The primary head's class count.
PRIMARY_CLASSES: int = 17

#: The secondary head's class count (17 tags + "none", DEF8).
SECONDARY_CLASSES: int = 18

#: The five binary attribute heads, in contract order.
ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "scheduled",
    "forward_looking",
    "primary_source",
    "quantified",
    "rumor",
)

#: The width of the network's output: 17 + 18 + 5.
OUTPUT_WIDTH: int = PRIMARY_CLASSES + SECONDARY_CLASSES + len(ATTRIBUTE_NAMES)

#: The pinned distilroberta-base facts Task 14 validates.
PINNED_FACTS: Dict[str, int] = {
    "num_hidden_layers": 6,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 514,
    "vocab_size": 50265,
    "pad_token_id": 1,
}

#: RoBERTa's special-token ids.
BOS_TOKEN_ID: int = 0
PAD_TOKEN_ID: int = 1
EOS_TOKEN_ID: int = 2

#: The content tokens that fit beside <s> and </s> in 514 positions.
MAX_CONTENT_TOKENS: int = PINNED_FACTS["max_position_embeddings"] - PAD_TOKEN_ID - 1 - 2

#: The maximum input length Task 15 truncates to.
DEFAULT_MAX_TOKENS: int = 512

#: The numeral-preserving budget of DEF18.
DEFAULT_NUMERAL_BUDGET: int = 96

#: Task 15's numeral-bearing sentence patterns.
NUMERAL_PATTERNS: Tuple[str, ...] = (
    r"\$[\d,.]+",
    r"\d+(?:\.\d+)?%",
    r"\d+(?:\.\d+)?\s*bps",
    r"\d{4,}",
    r"million|billion|trillion",
)

#: Task 15's sentence boundary.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

#: The parameter Task 14's inventory pins at zero rows but the checkpoint carries.
TOKEN_TYPE_PARAMETER: str = "embeddings.token_type_embeddings.weight"


# ----------------------------------------------------------------------------
# The network
# ----------------------------------------------------------------------------


class DistilRobertaStudent(torch.nn.Module):
    """
    The seven-head distilroberta-base student of Section 4.1.

        The backbone's ``embeddings`` and ``encoder`` are registered as this
        module's own children, so every parameter name equals the one in Task
        14's inventory (``embeddings.word_embeddings.weight``,
        ``encoder.layer.0.attention.self.query.weight``, ``primary_tag.weight``,
        ``rumor.bias`` and so on). The backbone object is kept as an
        unregistered reference only to run its forward pass, so no parameter is
        counted twice. There is no pooler: the pooled representation is the
        masked mean of the final hidden states (DEF17).

        Parameters
        ----------
        backbone : RobertaModel
            A RoBERTa encoder built without its pooling layer.
        seed : int, optional
            The seed of the heads' initialisation; the global torch RNG state is
            restored afterwards. Defaults to 14.

        Attributes
        ----------
        embeddings : torch.nn.Module
            The backbone's embedding block.
        encoder : torch.nn.Module
            The backbone's six transformer layers.
        primary_tag, secondary_tag : torch.nn.Linear
            The two categorical heads, 768 -> 17 and 768 -> 18.
        scheduled, forward_looking, primary_source, quantified, rumor : torch.nn.Linear
            The five binary heads, 768 -> 1 each.

        Raises
        ------
        ValueError
            If the backbone departs from a pinned distilroberta-base fact or
            carries a pooling layer.
    """

    def __init__(self, backbone: RobertaModel, seed: int = 14) -> None:
        super().__init__()
        config = backbone.config
        drift = {
            key: (getattr(config, key, None), value)
            for key, value in PINNED_FACTS.items()
            if getattr(config, key, None) != value
        }
        if drift:
            raise ValueError(
                f"the backbone departs from the pinned distilroberta-base facts "
                f"(declared, pinned): {drift}."
            )
        if getattr(backbone, "pooler", None) is not None:
            raise ValueError(
                "build the backbone with add_pooling_layer=False; the student "
                "pools by masked mean (DEF17)."
            )
        # Register the backbone's two blocks under Task 14's names.
        self.embeddings = backbone.embeddings
        self.encoder = backbone.encoder
        # Keep the backbone for its forward pass without registering it again.
        object.__setattr__(self, "_backbone", backbone)
        hidden = int(config.hidden_size)
        state = torch.random.get_rng_state()
        torch.manual_seed(int(seed) % (2 ** 31))
        try:
            self.primary_tag = torch.nn.Linear(hidden, PRIMARY_CLASSES)
            self.secondary_tag = torch.nn.Linear(hidden, SECONDARY_CLASSES)
            for name in ATTRIBUTE_NAMES:
                setattr(self, name, torch.nn.Linear(hidden, 1))
            std = float(getattr(config, "initializer_range", 0.02))
            for head in self._heads():
                torch.nn.init.normal_(head.weight, mean=0.0, std=std)
                torch.nn.init.zeros_(head.bias)
        finally:
            torch.random.set_rng_state(state)

    def _heads(self) -> List[torch.nn.Linear]:
        """
        Return the seven heads in output order.

                Returns
                -------
                List[torch.nn.Linear]
                    ``primary_tag``, ``secondary_tag`` and the five attribute heads.
        """
        return [self.primary_tag, self.secondary_tag] + [
            getattr(self, name) for name in ATTRIBUTE_NAMES
        ]

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch and return its masked-mean representation.

                The content tokens are the positions ``attention_mask`` marks, in
                order; the pad id is ignored, because Tasks 15 and 17 pad with 0,
                which RoBERTa reads as <s>. Each row is rebuilt as
                ``<s> content[:510] </s>``, padded with RoBERTa's pad id, and given
                explicit position ids starting after the padding offset.

                Parameters
                ----------
                input_ids : torch.Tensor
                    Content token ids without special tokens, of shape
                    (batch, tokens).
                attention_mask : torch.Tensor
                    1 on content tokens and 0 on padding, of shape (batch, tokens).

                Returns
                -------
                torch.Tensor
                    The pooled representation of shape (batch, 768).

                Raises
                ------
                ValueError
                    If the two tensors' shapes disagree.
        """
        if input_ids.shape != attention_mask.shape or input_ids.dim() != 2:
            raise ValueError(
                "input_ids and attention_mask must share one (batch, tokens) "
                f"shape; received {tuple(input_ids.shape)} and "
                f"{tuple(attention_mask.shape)}."
            )
        device = input_ids.device
        mask = attention_mask.to(torch.bool)
        batch = input_ids.shape[0]
        lengths = mask.sum(dim=1).clamp(max=MAX_CONTENT_TOKENS)
        width = int(lengths.max().item()) + 2 if batch else 2
        # Move each row's content tokens to the front, in their original order.
        order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
        content = torch.gather(input_ids.long(), 1, order)[:, : width - 2]
        steps = torch.arange(width, device=device)
        ids = torch.full((batch, width), PAD_TOKEN_ID, dtype=torch.long, device=device)
        ids[:, 0] = BOS_TOKEN_ID
        keep = steps[: width - 2].unsqueeze(0) < lengths.unsqueeze(1)
        ids[:, 1 : width - 1] = torch.where(keep, content, torch.full_like(content, PAD_TOKEN_ID))
        ids[torch.arange(batch, device=device), lengths + 1] = EOS_TOKEN_ID
        real = steps.unsqueeze(0) < (lengths + 2).unsqueeze(1)
        positions = torch.where(
            real,
            steps.unsqueeze(0) + PAD_TOKEN_ID + 1,
            torch.full_like(ids, PAD_TOKEN_ID),
        )
        hidden = self._backbone(
            input_ids=ids,
            attention_mask=real.long(),
            position_ids=positions,
        ).last_hidden_state
        weights = real.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Return the student's forty logits per article.

                Parameters
                ----------
                input_ids : torch.Tensor
                    Content token ids without special tokens, of shape
                    (batch, tokens).
                attention_mask : torch.Tensor
                    1 on content tokens and 0 on padding, of shape (batch, tokens).

                Returns
                -------
                torch.Tensor
                    Logits of shape (batch, 40): the 17 primary, the 18 secondary and
                    the 5 attribute logits, the layout Task 15 slices.
        """
        pooled = self.pooled(input_ids, attention_mask)
        return torch.cat([head(pooled) for head in self._heads()], dim=1)


def distilroberta_config(**overrides: Any) -> RobertaConfig:
    """
    Return distilroberta-base's configuration without downloading it.

        Useful for building an untrained network of the exact architecture,
        for tests or for loading weights from a local state dict.

        Parameters
        ----------
        **overrides : Any
            Fields to override, such as ``hidden_dropout_prob``.

        Returns
        -------
        RobertaConfig
            The configuration: the pinned facts, ``type_vocab_size=1`` as in the
            published checkpoint, layer-norm epsilon 1e-5 and RoBERTa's special
            token ids.
    """
    values: Dict[str, Any] = {
        **PINNED_FACTS,
        "type_vocab_size": 1,
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "layer_norm_eps": 1e-5,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
    }
    values.update(overrides)
    return RobertaConfig(**values)


def build_student_network(
    pretrained: Optional[Union[str, Path]] = BACKBONE_NAME,
    *,
    seed: int = 14,
    local_files_only: bool = False,
    **config_overrides: Any,
) -> DistilRobertaStudent:
    """
    Build the student (service ``student_network``) on the pretrained backbone.

        Parameters
        ----------
        pretrained : Optional[Union[str, Path]], optional
            A Hugging Face identifier or a local directory holding the
            distilroberta-base checkpoint, or ``None`` for an untrained backbone
            of the same architecture. Defaults to :data:`BACKBONE_NAME`.
        seed : int, optional
            The heads' initialisation seed. Keyword-only; defaults to 14.
        local_files_only : bool, optional
            Refuse to download. Keyword-only; defaults to False.
        **config_overrides : Any
            Configuration fields to override, such as dropout rates.

        Returns
        -------
        DistilRobertaStudent
            The untrained student, on the CPU.

        Raises
        ------
        ValueError
            If the backbone departs from a pinned distilroberta-base fact.
    """
    if pretrained is None:
        backbone = RobertaModel(distilroberta_config(**config_overrides), add_pooling_layer=False)
    else:
        backbone = RobertaModel.from_pretrained(
            str(pretrained),
            add_pooling_layer=False,
            local_files_only=local_files_only,
            **config_overrides,
        )
    return DistilRobertaStudent(backbone, seed=seed)


def checkpoint_files(pretrained: Union[str, Path] = BACKBONE_NAME) -> List[str]:
    """
    Return the backbone checkpoint files Task 14 hashes (``student_checkpoint_paths``).

        Parameters
        ----------
        pretrained : Union[str, Path], optional
            A local checkpoint directory, or a Hugging Face identifier, which is
            resolved through the local cache (downloading it if needed).
            Defaults to :data:`BACKBONE_NAME`.

        Returns
        -------
        List[str]
            The sorted weight and configuration files: ``*.safetensors``,
            ``pytorch_model.bin`` and ``config.json``, whichever exist.

        Raises
        ------
        FileNotFoundError
            If the directory holds no weight file.
    """
    directory = Path(pretrained)
    if not directory.is_dir():
        from huggingface_hub import snapshot_download

        directory = Path(
            snapshot_download(
                str(pretrained),
                allow_patterns=["*.safetensors", "pytorch_model.bin", "config.json"],
            )
        )
    files = sorted(
        str(path)
        for pattern in ("*.safetensors", "pytorch_model.bin", "config.json")
        for path in directory.glob(pattern)
    )
    if not any(not name.endswith("config.json") for name in files):
        raise FileNotFoundError(f"no checkpoint weights under {directory}.")
    return files


# ----------------------------------------------------------------------------
# Task 14 conformance
# ----------------------------------------------------------------------------


def check_against_task14_inventory(
    model: torch.nn.Module,
    inventory: Mapping[str, Tuple[int, ...]],
    *,
    allow_token_type_row: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Compare a network's parameters with Task 14's parameter inventory.

        Every parameter name and shape of the network is compared with the
        inventory ``build_student_parameter_inventory_task_14`` returns. The one
        expected difference is the checkpoint's 1 x 768 token-type table, which
        Task 14 pins at zero rows; it is reported under ``known_deviations``.

        Parameters
        ----------
        model : torch.nn.Module
            The network, normally a :class:`DistilRobertaStudent`.
        inventory : Mapping[str, Tuple[int, ...]]
            Task 14's parameter shapes keyed by name.
        allow_token_type_row : bool, optional
            Accept the token-type deviation. Keyword-only; defaults to True.
        strict : bool, optional
            Raise when any other difference is found. Keyword-only; defaults to
            True.

        Returns
        -------
        Dict[str, Any]
            ``matches`` (bool), ``n_params`` (the network's count),
            ``n_params_inventory`` (the inventory's count), ``missing``,
            ``unexpected``, ``shape_mismatches`` and ``known_deviations``.

        Raises
        ------
        ValueError
            If ``strict`` and the network departs from the inventory beyond the
            accepted deviation.
    """
    actual = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
    expected = {name: tuple(int(size) for size in shape) for name, shape in inventory.items()}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatches = {
        name: {"network": actual[name], "inventory": expected[name]}
        for name in sorted(set(actual) & set(expected))
        if actual[name] != expected[name]
    }
    known: Dict[str, Any] = {}
    if allow_token_type_row and TOKEN_TYPE_PARAMETER in mismatches:
        shapes = mismatches[TOKEN_TYPE_PARAMETER]
        if shapes["inventory"][0] == 0 and shapes["network"][1:] == shapes["inventory"][1:]:
            known[TOKEN_TYPE_PARAMETER] = {
                **shapes,
                "reason": "distilroberta-base has type_vocab_size=1; Task 14 pins 0",
            }
            del mismatches[TOKEN_TYPE_PARAMETER]
    report = {
        "matches": not (missing or unexpected or mismatches),
        "n_params": int(sum(int(np.prod(shape)) for shape in actual.values())),
        "n_params_inventory": int(sum(int(np.prod(shape)) for shape in expected.values())),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": mismatches,
        "known_deviations": known,
    }
    if strict and not report["matches"]:
        raise ValueError(f"the network departs from Task 14's inventory: {report}.")
    return report


# ----------------------------------------------------------------------------
# Tokenisation and truncation (mirrors Task 15)
# ----------------------------------------------------------------------------


class StudentCodec:
    """
    The student's tokeniser pair: ``encode`` and ``decode`` without special tokens.

        Task 15 encodes the article, then each numeral-bearing sentence, and
        concatenates the ids, so special tokens must be added once by the
        network, never by ``encode``. The tokenizer's length warning is silenced
        because truncation is Task 15's job.

        Parameters
        ----------
        tokenizer : Any
            A Hugging Face tokenizer of distilroberta-base (byte-level BPE).

        Raises
        ------
        ValueError
            If the tokenizer's vocabulary or special-token ids differ from
            distilroberta-base's.
    """

    def __init__(self, tokenizer: Any) -> None:
        if len(tokenizer) != PINNED_FACTS["vocab_size"]:
            raise ValueError(
                f"the tokenizer has {len(tokenizer)} tokens; distilroberta-base "
                f"has {PINNED_FACTS['vocab_size']}."
            )
        specials = (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id)
        if specials != (BOS_TOKEN_ID, PAD_TOKEN_ID, EOS_TOKEN_ID):
            raise ValueError(f"unexpected special-token ids {specials}.")
        tokenizer.model_max_length = 10 ** 9
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        """
        Encode a text as byte-level BPE ids, with no special tokens.

                Parameters
                ----------
                text : str
                    Any text.

                Returns
                -------
                List[int]
                    The token ids.
        """
        return list(self.tokenizer(str(text), add_special_tokens=False)["input_ids"])

    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decode ids back to text (the inverse Task 15's numeral scan reads).

                Parameters
                ----------
                tokens : Sequence[int]
                    Token ids.

                Returns
                -------
                str
                    The text.
        """
        return str(self.tokenizer.decode([int(token) for token in tokens]))


def load_codec(pretrained: Union[str, Path] = BACKBONE_NAME, *, local_files_only: bool = False) -> StudentCodec:
    """
    Load distilroberta-base's tokenizer as a :class:`StudentCodec`.

        Parameters
        ----------
        pretrained : Union[str, Path], optional
            A Hugging Face identifier or a local directory. Defaults to
            :data:`BACKBONE_NAME`.
        local_files_only : bool, optional
            Refuse to download. Keyword-only; defaults to False.

        Returns
        -------
        StudentCodec
            The codec.
    """
    from transformers import AutoTokenizer

    return StudentCodec(
        AutoTokenizer.from_pretrained(str(pretrained), local_files_only=local_files_only)
    )


def extract_numeral_sentences(body: str) -> List[str]:
    """
    Return the numeral-bearing sentences of a text, as Task 15 does.

        Parameters
        ----------
        body : str
            The text to scan.

        Returns
        -------
        List[str]
            The sentences matching at least one of :data:`NUMERAL_PATTERNS`,
            in reading order; an empty list for a blank text.

        Examples
        --------
        >>> extract_numeral_sentences("Sales rose 12%. Shares fell. It paid $4.5bn.")
        ['Sales rose 12%.', 'It paid $4.5bn.']
    """
    if body.strip() == "":
        return []
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(body) if part.strip() != ""]
    return [
        sentence
        for sentence in sentences
        if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in NUMERAL_PATTERNS)
    ]


def build_student_input(
    text: str,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    numeral_budget: int = DEFAULT_NUMERAL_BUDGET,
) -> List[int]:
    """
    Apply Task 15's numeral-preserving truncation (DEF18) to one text.

        Mirrors ``build_student_input_task_15`` on its composed document
        ``title + " " + body``: the head keeps ``max_tokens - numeral_budget``
        tokens, the numeral-bearing sentences of the truncated tail are appended
        in reading order while they fit the budget, and any unused budget is
        filled from the start of the tail. A blank text yields no tokens.

        Parameters
        ----------
        text : str
            The composed document.
        encode : Callable[[str], Sequence[int]]
            The tokeniser.
        decode : Callable[[Sequence[int]], str]
            The inverse tokeniser.
        max_tokens : int, optional
            The maximum input length. Defaults to 512.
        numeral_budget : int, optional
            The reserved numeral budget. Defaults to 96.

        Returns
        -------
        List[int]
            The truncated token ids.

        Examples
        --------
        >>> words = "a b c d e f g h. It rose 5%.".split()
        >>> enc = lambda t: [words.index(w) for w in t.split()]
        >>> dec = lambda ids: " ".join(words[i] for i in ids)
        >>> build_student_input(" ".join(words), enc, dec, max_tokens=6, numeral_budget=3)
        [0, 1, 2, 8, 9, 10]
    """
    if text.strip() == "":
        return []
    tokens = [int(value) for value in encode(text)]
    head = tokens[: int(max_tokens) - int(numeral_budget)]
    tail = tokens[int(max_tokens) - int(numeral_budget):]
    tail_text = decode(tail) if tail else ""
    used: List[List[int]] = []
    numeral_tokens = 0
    for sentence in extract_numeral_sentences(tail_text):
        sentence_tokens = [int(value) for value in encode(sentence)]
        if numeral_tokens + len(sentence_tokens) > int(numeral_budget):
            break
        used.append(sentence_tokens)
        numeral_tokens += len(sentence_tokens)
    ids = list(head)
    for sentence_tokens in used:
        ids.extend(sentence_tokens)
    ids.extend(tail[: int(numeral_budget) - numeral_tokens])
    return ids


# ----------------------------------------------------------------------------
# The scorers
# ----------------------------------------------------------------------------


class StudentScorer:
    """
    The four student services around one shared :class:`DistilRobertaStudent`.

        Tasks 15 and 17 train ``student_network`` in place (Task 17 loads the v1
        weights into the same module), so scorers that read this one module see
        the v1 weights in Task 16 and the v2 weights in Tasks 17 and 18 without
        any reloading.

        Parameters
        ----------
        model : torch.nn.Module
            The student, normally a :class:`DistilRobertaStudent`.
        codec : Any
            An object with ``encode(text)`` and ``decode(ids)``, normally a
            :class:`StudentCodec`.
        batch_size : int, optional
            The inference minibatch size. Defaults to 64.
        max_tokens : int, optional
            The truncation length. Defaults to 512.
        numeral_budget : int, optional
            The numeral budget. Defaults to 96.

        Raises
        ------
        TypeError
            If ``model`` is not a ``torch.nn.Module`` or the codec lacks
            ``encode`` or ``decode``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        codec: Any,
        *,
        batch_size: int = 64,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        numeral_budget: int = DEFAULT_NUMERAL_BUDGET,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model must be a torch.nn.Module; received {type(model).__name__}.")
        if not (callable(getattr(codec, "encode", None)) and callable(getattr(codec, "decode", None))):
            raise TypeError("codec must provide encode(text) and decode(ids).")
        self.model = model
        self.codec = codec
        self.batch_size = max(1, int(batch_size))
        self.max_tokens = int(max_tokens)
        self.numeral_budget = int(numeral_budget)

    # ------------------------------------------------------------------
    # The services
    # ------------------------------------------------------------------

    def student_model(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        """
        Score texts with the first-pass student (service ``student_model``).

                Serves Task 16, which sends DEF16's ``title + " " + body`` per
                article.

                Parameters
                ----------
                texts : Sequence[str]
                    One composed document per article.

                Returns
                -------
                Dict[str, np.ndarray]
                    ``primary_probs`` of shape (n, 17), and ``head_outputs`` of shape
                    (n, 7): the primary and secondary argmax indices and the five
                    attribute decisions at 0.5, in the stand-in's layout.
        """
        logits = self.logits(texts)
        primary = _softmax(logits[:, :PRIMARY_CLASSES])
        secondary = logits[:, PRIMARY_CLASSES:PRIMARY_CLASSES + SECONDARY_CLASSES]
        attributes = _sigmoid(logits[:, PRIMARY_CLASSES + SECONDARY_CLASSES:])
        heads = np.column_stack(
            [
                primary.argmax(axis=1).astype(np.float64),
                secondary.argmax(axis=1).astype(np.float64),
                (attributes >= 0.5).astype(np.float64),
            ]
        ) if len(logits) else np.zeros((0, 2 + len(ATTRIBUTE_NAMES)))
        return {"primary_probs": primary, "head_outputs": heads}

    def student_predict_fn(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Score a frame with the retrained student (service ``student_predict_fn``).

                Serves Tasks 17 and 18. The text is the frame's ``text`` column when
                present, else ``title + " " + body`` as Task 15 composes it.

                Parameters
                ----------
                frame : pd.DataFrame
                    Rows carrying ``text``, or ``title`` and ``body``.

                Returns
                -------
                pd.DataFrame
                    ``head_tag_probs`` (17 floats), ``head_secondary_tag_probs`` (18
                    floats) and ``head_attr_<name>_prob`` per attribute, indexed like
                    ``frame``.

                Raises
                ------
                TypeError
                    If ``frame`` is not a DataFrame.
                ValueError
                    If the frame carries neither text layout.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a DataFrame.")
        logits = self.logits(_compose_texts(frame))
        primary = _softmax(logits[:, :PRIMARY_CLASSES])
        secondary = _softmax(logits[:, PRIMARY_CLASSES:PRIMARY_CLASSES + SECONDARY_CLASSES])
        attributes = _sigmoid(logits[:, PRIMARY_CLASSES + SECONDARY_CLASSES:])
        columns: Dict[str, Any] = {
            "head_tag_probs": [row.tolist() for row in primary],
            "head_secondary_tag_probs": [row.tolist() for row in secondary],
        }
        for slot, name in enumerate(ATTRIBUTE_NAMES):
            columns[f"head_attr_{name}_prob"] = attributes[:, slot].astype(float).tolist()
        return pd.DataFrame(columns, index=frame.index)

    def student_encode(self, text: str) -> List[int]:
        """
        Encode a text for Task 17's retrain (service ``student_encode``).

                Parameters
                ----------
                text : str
                    Any text.

                Returns
                -------
                List[int]
                    The token ids, with no special tokens.
        """
        return [int(token) for token in self.codec.encode(text)]

    def services(self) -> Dict[str, Any]:
        """
        Return the four entries to merge into the pipeline bundle's services.

                Returns
                -------
                Dict[str, Any]
                    ``student_network`` (the module), ``student_model``,
                    ``student_predict_fn`` and ``student_encode``.
        """
        return {
            "student_network": self.model,
            "student_model": self.student_model,
            "student_predict_fn": self.student_predict_fn,
            "student_encode": self.student_encode,
        }

    def training_config_entries(self) -> Dict[str, Any]:
        """
        Return the codec entries of Task 15's ``student_training_config``.

                Returns
                -------
                Dict[str, Any]
                    ``encode``, ``decode`` and RoBERTa's ``pad_token_id`` (1). The pad
                    id is cosmetic: the network rebuilds rows from the mask.
        """
        return {
            "encode": self.student_encode,
            "decode": self.codec.decode,
            "pad_token_id": PAD_TOKEN_ID,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return the forty logits of each text, in input order.

                Texts are truncated by Task 15's rule, sorted by length into
                minibatches to limit padding, and scored in evaluation mode under
                ``torch.inference_mode``; the module's training mode is restored
                afterwards, because Tasks 15 and 17 train the same module.

                Parameters
                ----------
                texts : Sequence[str]
                    Composed documents.

                Returns
                -------
                np.ndarray
                    Float64 logits of shape (n, 40).

                Raises
                ------
                ValueError
                    If the model returns logits of the wrong shape.
        """
        rows = [
            build_student_input(
                str(text), self.student_encode, self.codec.decode,
                self.max_tokens, self.numeral_budget,
            )
            for text in texts
        ]
        out = np.zeros((len(rows), OUTPUT_WIDTH), dtype=np.float64)
        if not rows:
            return out
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        order = sorted(range(len(rows)), key=lambda position: len(rows[position]))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(order), self.batch_size):
                    positions = order[start:start + self.batch_size]
                    width = max(1, max(len(rows[position]) for position in positions))
                    ids = torch.full((len(positions), width), PAD_TOKEN_ID, dtype=torch.long)
                    mask = torch.zeros((len(positions), width), dtype=torch.long)
                    for slot, position in enumerate(positions):
                        length = len(rows[position])
                        if length:
                            ids[slot, :length] = torch.as_tensor(rows[position], dtype=torch.long)
                            mask[slot, :length] = 1
                    batch_logits = self.model(ids.to(device), mask.to(device))
                    if tuple(batch_logits.shape) != (len(positions), OUTPUT_WIDTH):
                        raise ValueError(
                            f"the model returned logits of shape {tuple(batch_logits.shape)}; "
                            f"expected ({len(positions)}, {OUTPUT_WIDTH})."
                        )
                    out[positions] = batch_logits.float().cpu().numpy()
        finally:
            self.model.train(was_training)
        return out


def make_student_services(
    pretrained: Optional[Union[str, Path]] = BACKBONE_NAME,
    *,
    seed: int = 14,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 64,
    local_files_only: bool = False,
    codec: Any = None,
) -> StudentScorer:
    """
    Build the student, its codec and its scorers in one call.

        Parameters
        ----------
        pretrained : Optional[Union[str, Path]], optional
            The checkpoint, as for :func:`build_student_network`. Defaults to
            :data:`BACKBONE_NAME`.
        seed : int, optional
            The heads' initialisation seed. Keyword-only; defaults to 14.
        device : Optional[Union[str, torch.device]], optional
            Where to place the network. Keyword-only; defaults to ``None``,
            which picks CUDA, then Apple MPS, then the CPU.
        batch_size : int, optional
            The inference minibatch size. Keyword-only; defaults to 64.
        local_files_only : bool, optional
            Refuse to download. Keyword-only; defaults to False.
        codec : Any, optional
            A codec to use instead of loading the tokenizer. Keyword-only;
            defaults to ``None``.

        Returns
        -------
        StudentScorer
            The scorer, whose :meth:`StudentScorer.services` and
            :meth:`StudentScorer.training_config_entries` feed the bundle.
    """
    model = build_student_network(pretrained, seed=seed, local_files_only=local_files_only)
    model.to(_resolve_device(device))
    if codec is None:
        codec = load_codec(pretrained if pretrained is not None else BACKBONE_NAME,
                           local_files_only=local_files_only)
    return StudentScorer(model, codec, batch_size=batch_size)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _compose_texts(frame: pd.DataFrame) -> List[str]:
    """
    Compose each row's document as Tasks 15 and 16 do.

        Parameters
        ----------
        frame : pd.DataFrame
            Rows carrying ``text``, or ``title`` and ``body``.

        Returns
        -------
        List[str]
            The ``text`` column as stored, else ``title + " " + body`` with
            missing parts rendered empty.

        Raises
        ------
        ValueError
            If the frame carries neither layout.
    """
    if "text" in frame.columns:
        return frame["text"].fillna("").astype(str).tolist()
    if {"title", "body"}.issubset(frame.columns):
        titles = frame["title"].fillna("").astype(str)
        bodies = frame["body"].fillna("").astype(str)
        return (titles + " " + bodies).tolist()
    raise ValueError("frame must carry text, or title and body.")


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """
    Pick the inference device.

        Parameters
        ----------
        device : Optional[Union[str, torch.device]]
            An explicit device, or ``None``.

        Returns
        -------
        torch.device
            The explicit device, else CUDA, then Apple MPS, then the CPU.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Return the row-wise softmax.

        Parameters
        ----------
        logits : np.ndarray
            A (n, k) array.

        Returns
        -------
        np.ndarray
            Rows summing to one.

        Examples
        --------
        >>> _softmax(np.array([[0.0, 0.0]])).tolist()
        [[0.5, 0.5]]
    """
    shifted = logits - logits.max(axis=1, keepdims=True) if len(logits) else logits
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True) if len(logits) else exp


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """
    Return the element-wise logistic function.

        Parameters
        ----------
        logits : np.ndarray
            Any array.

        Returns
        -------
        np.ndarray
            Values in (0, 1).

        Examples
        --------
        >>> _sigmoid(np.array([0.0])).tolist()
        [0.5]
    """
    return 1.0 / (1.0 + np.exp(-logits))


__all__ = [
    "ATTRIBUTE_NAMES",
    "BACKBONE_NAME",
    "DistilRobertaStudent",
    "OUTPUT_WIDTH",
    "StudentCodec",
    "StudentScorer",
    "build_student_input",
    "build_student_network",
    "check_against_task14_inventory",
    "checkpoint_files",
    "distilroberta_config",
    "extract_numeral_sentences",
    "load_codec",
    "make_student_services",
]
