#!/usr/bin/env python3
# ============================================================================#
#
#  Buy the Rumor, Sell the News: When Is News Priced In?
#  --  azure_teacher_clients.py
#
#  Live Azure OpenAI replacements for the three offline teacher stand-ins of
#  the usage example (Step 3.2): ``teacher_client`` (Task 13),
#  ``teacher_call`` (Tasks 16 and 17) and ``teacher_batch_client``
#  (Task 18). They call the study's teacher, gpt-5-mini on Azure OpenAI
#  (Section 4.1), under the frozen Task 12 prompt and response schema.
#
#  The notebook itself never imports an LLM SDK: it reaches the teacher only
#  through these injected services. This module is that bridge, and it keeps
#  the notebook's contracts:
#  • Same prompt. The system message is Task 12's frozen template and the
#    user message is the article suffix truncated by Task 13's rule, so the
#    ID-only services (Tasks 16-18) send exactly what Task 13 would.
#  • Same provenance. The constructor refuses a template whose SHA-256 does
#    not equal its prompt_version, and every call refuses a different
#    prompt_version, model or system prompt.
#  • Same retry semantics. Transport failures (rate limits, 5xx, timeouts,
#    dropped connections) are retried here with backoff, and if they persist
#    they surface as the built-in ConnectionError / TimeoutError /
#    RuntimeError classes that the notebook's retry loops catch. The openai
#    SDK's own exception classes are NOT subclasses of those, so without the
#    translation a single HTTP 429 would abort a stage. Configuration errors
#    (bad key, unknown deployment, malformed request) propagate unchanged and
#    halt the run, as the notebook requires of programming errors.
#  • Same label contract. Replies are checked against the 17-tag enum, the
#    "none" literal, the five booleans and the secondary != primary rule
#    before any service returns them. Task 17 raises on an off-taxonomy tag,
#    so an unchecked reply could halt the pipeline.
#  • No double charge. A label is bought at most once per
#    (prompt_version, article_id), optionally persisted to a JSONL file, so
#    a Task 18 batch retry or a rerun never re-buys a label.
#
#  gpt-5-mini is a reasoning model, and that has three consequences that
#  the notebook's DEF14 pins cannot express:
#  • The API rejects temperature / top_p other than their defaults, so
#    they are NOT sent by default (send_sampling_params=False) and a
#    RuntimeWarning records the deviation from the pinned 0.0 / 1.0.
#  • The output cap is sent as max_completion_tokens, which also covers
#    hidden reasoning tokens. With the pinned 200-token cap, reasoning
#    must be kept short (reasoning_effort="minimal", the default) or the
#    reply comes back empty with finish_reason="length".
#  • No seed is exposed (as the notebook already records).
#
#  Author: CS Chirinda
#  License: MIT
#
# ============================================================================#

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd

try:
    import openai
except ImportError as _error:  # pragma: no cover - exercised only without openai
    raise ImportError(
        "azure_teacher_clients requires the openai package: "
        "python -m pip install openai"
    ) from _error


# ----------------------------------------------------------------------------
# The frozen label contract (mirrors Tasks 12, 13 and 16)
# ----------------------------------------------------------------------------

#: The five binary attributes, in contract order.
ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "scheduled",
    "forward_looking",
    "primary_source",
    "quantified",
    "rumor",
)

#: The seven fields of the strict response schema.
CONTRACT_FIELDS: Tuple[str, ...] = ("primary_tag", "secondary_tag") + ATTRIBUTE_NAMES

#: The literal secondary-tag choice that means "no second event".
NONE_LABEL: str = "none"

#: The study's teacher model (Task 12's TEACHER_MODEL_NAME).
DEFAULT_MODEL_NAME: str = "gpt-5-mini"

#: The environment-variable names of llm_api.teacher in config.yaml.
DEFAULT_ENV_VARS: Dict[str, str] = {
    "api_key_env_var": "AZURE_OPENAI_API_KEY",
    "endpoint_env_var": "AZURE_OPENAI_ENDPOINT",
    "api_version_env_var": "AZURE_OPENAI_API_VERSION",
    "deployment_name_env_var": "AZURE_OPENAI_DEPLOYMENT",
}

#: HTTP statuses that are transient and worth retrying on the same request.
RETRYABLE_STATUS: Tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)


class TeacherConfigurationError(ValueError):
    """A provenance or configuration mismatch that must halt the run.

    It derives from ValueError, which none of the notebook's retry loops
    catch, so the stage fails instead of silently relabelling under a
    drifted prompt.
    """


# ----------------------------------------------------------------------------
# Token counting and truncation (mirrors Task 13's _compose_teacher_suffix)
# ----------------------------------------------------------------------------


def whitespace_token_count(text: str) -> int:
    """Count whitespace tokens: the notebook's default counter (Task 12/13)."""
    return len(text.split())


def tiktoken_counter(encoding_name: str = "o200k_base") -> Callable[[str], int]:
    """Return a BPE token counter for the gpt-5 family's tokenizer.

    Only use it if Task 13 receives the same counter (``count_tokens``);
    otherwise Tasks 13 and 16-18 would truncate articles differently.
    """
    import tiktoken  # optional dependency

    encoding = tiktoken.get_encoding(encoding_name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def compose_article_suffix(
    title: str,
    body: str,
    prompt_prefix: str,
    max_output_tokens: int,
    max_context_length_tokens: Optional[int],
    count_tokens: Callable[[str], int] = whitespace_token_count,
) -> Tuple[str, bool]:
    """Join and truncate the article exactly as Task 13 does.

    The suffix is ``title + "\\n\\n" + body`` (a blank part is dropped with the
    separator), and the longest character prefix whose token count fits
    ``window - max_output_tokens - count(prompt_prefix)`` is kept.

    Returns ``(text, truncated)``. Raises ValueError if the budget is
    non-positive.
    """
    title_blank = title.strip() == ""
    body_blank = body.strip() == ""
    if title_blank and body_blank:
        # Task 13 sends an empty suffix and lets the teacher's reply decide.
        return "", False
    if title_blank:
        candidate = body
    elif body_blank:
        candidate = title
    else:
        candidate = title + "\n\n" + body
    if max_context_length_tokens is None:
        return candidate, False
    budget = int(max_context_length_tokens) - int(max_output_tokens) - int(
        count_tokens(prompt_prefix)
    )
    if budget <= 0:
        raise ValueError(
            f"the remaining context budget is {budget}: window "
            f"{max_context_length_tokens}, output allowance {max_output_tokens}."
        )
    if count_tokens(candidate) <= budget:
        return candidate, False
    low, high, best = 0, len(candidate), 0
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(candidate[:middle]) <= budget:
            best, low = middle, middle + 1
        else:
            high = middle - 1
    text = candidate[:best]
    return text, len(text) < len(candidate)


# ----------------------------------------------------------------------------
# Schema and reply validation
# ----------------------------------------------------------------------------


def api_response_schema(task12_schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Reduce Task 12's schema to the subset Azure strict structured outputs accept.

    Strict mode supports type / properties / enum / required /
    additionalProperties but not the ``allOf`` if/then branches Task 12 uses
    to forbid ``secondary_tag == primary_tag``, nor ``x_*`` metadata. That one
    rule is therefore enforced locally by :func:`validate_label`.
    """
    properties = task12_schema.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != set(CONTRACT_FIELDS):
        raise TeacherConfigurationError(
            "the Task 12 schema must declare exactly the seven contract fields "
            f"{list(CONTRACT_FIELDS)}."
        )
    return {
        "type": "object",
        "properties": {
            name: {key: value for key, value in dict(properties[name]).items()
                   if key in ("type", "enum", "description")}
            for name in CONTRACT_FIELDS
        },
        "required": list(CONTRACT_FIELDS),
        "additionalProperties": False,
    }


def validate_label(
    reply: Any, taxonomy: Tuple[str, ...]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Check one teacher reply against the frozen seven-field contract.

    Accepts a JSON string or a mapping. Returns ``(label, None)`` with
    ``secondary_tag`` normalised to None when the teacher reports "none", or
    ``(None, reason)`` when the reply is unusable.
    """
    if isinstance(reply, (str, bytes)):
        try:
            reply = json.loads(reply)
        except (TypeError, ValueError) as error:
            return None, f"unparseable JSON: {error}"
    if not isinstance(reply, Mapping):
        return None, f"reply is a {type(reply).__name__}, not a JSON object"
    extra = sorted(set(reply) - set(CONTRACT_FIELDS))
    if extra:
        return None, f"undeclared fields {extra}"
    missing = [name for name in CONTRACT_FIELDS if name not in reply]
    if missing:
        return None, f"missing fields {missing}"
    primary = reply["primary_tag"]
    if not isinstance(primary, str) or primary not in taxonomy:
        return None, f"primary_tag {primary!r} is outside the taxonomy"
    secondary = reply["secondary_tag"]
    if secondary is None or (isinstance(secondary, str) and secondary == NONE_LABEL):
        secondary = None
    elif not isinstance(secondary, str) or secondary not in taxonomy:
        return None, f"secondary_tag {secondary!r} is outside the taxonomy"
    elif secondary == primary:
        return None, f"secondary_tag repeats primary_tag {primary!r}"
    label: Dict[str, Any] = {"primary_tag": primary, "secondary_tag": secondary}
    for name in ATTRIBUTE_NAMES:
        value = reply[name]
        if not isinstance(value, bool):
            return None, f"{name} is {value!r}, not a boolean"
        label[name] = value
    return label, None


# ----------------------------------------------------------------------------
# Client construction
# ----------------------------------------------------------------------------


def _teacher_env_names(study_config: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Read the env-var names from llm_api.teacher, falling back to the defaults."""
    names = dict(DEFAULT_ENV_VARS)
    block = ((study_config or {}).get("llm_api") or {}).get("teacher") or {}
    for key in names:
        if block.get(key):
            names[key] = str(block[key])
    return names


def resolve_deployment(study_config: Optional[Mapping[str, Any]] = None) -> str:
    """Return the Azure deployment name from its environment variable."""
    variable = _teacher_env_names(study_config)["deployment_name_env_var"]
    deployment = os.environ.get(variable)
    if not deployment:
        raise TeacherConfigurationError(f"environment variable {variable} is not set.")
    return deployment


def build_azure_client(
    study_config: Optional[Mapping[str, Any]] = None,
    *,
    timeout: float = 60.0,
) -> "openai.AzureOpenAI":
    """Build an AzureOpenAI client from the environment variables config.yaml names.

    The key is read from the environment and never stored in the config (the
    study's ``api_key: null``). The SDK's own retries are switched off
    (``max_retries=0``) so that :class:`AzureTeacher` sees and counts every
    attempt.
    """
    names = _teacher_env_names(study_config)
    values = {}
    for key in ("api_key_env_var", "endpoint_env_var", "api_version_env_var"):
        value = os.environ.get(names[key])
        if not value:
            raise TeacherConfigurationError(
                f"environment variable {names[key]} is not set."
            )
        values[key] = value
    return openai.AzureOpenAI(
        api_key=values["api_key_env_var"],
        azure_endpoint=values["endpoint_env_var"],
        api_version=values["api_version_env_var"],
        timeout=timeout,
        max_retries=0,
    )


# ----------------------------------------------------------------------------
# The teacher
# ----------------------------------------------------------------------------

ArticleSource = Union[pd.DataFrame, str, os.PathLike, Callable[[], pd.DataFrame]]


class AzureTeacher:
    """The study's gpt-5-mini teacher on Azure OpenAI, behind the notebook's three service shapes.

    Parameters
    ----------
    prompt_template, response_schema, decoding_config : Mapping
        Task 12's three payloads, i.e. the return values of
        ``build_teacher_prompt_template_task_12(STUDY_CONFIG)``,
        ``build_teacher_response_schema_task_12(STUDY_CONFIG)`` and
        ``build_teacher_decoding_config_task_12(STUDY_CONFIG)``. They are pure
        functions of the configuration, so they equal what stage 12 produces.
    articles : DataFrame, path or zero-argument callable
        The corpus the ID-only services resolve text from: the cleansed
        ``retained_articles_c`` that Task 13 labels. A path (Parquet or
        pickle) or callable is read lazily on the first ID-only call, so it
        can point at ``<output_dir>/stage_outputs/retained_articles_c.parquet``,
        which stage 3 writes before any teacher stage runs.
    client : optional
        An ``openai.AzureOpenAI`` (or compatible) client. Defaults to
        :func:`build_azure_client`.
    deployment : str, optional
        The Azure deployment name. Defaults to ``$AZURE_OPENAI_DEPLOYMENT``.
    count_tokens : callable, optional
        The truncation counter. Must be the one Task 13 uses; the pipeline
        passes none, so the default is the whitespace counter.
    reasoning_effort : str or None
        Sent to the API. "minimal" keeps reasoning inside the 200-token cap.
    send_sampling_params : bool
        Send the pinned temperature/top_p. gpt-5 models reject them.
    response_format : "json_schema" or "json_object"
        Strict structured outputs (default, per ``teacher.output_format``) or
        plain JSON mode.
    max_transport_retries : int
        Retries of one request on 429 / 5xx / timeouts before giving up.
    per_article_attempts : int
        Logical attempts per article inside ``teacher_batch_client``.
    max_workers : int
        Concurrent requests inside ``teacher_batch_client``.
    label_cache_path : path, optional
        JSONL file of purchased labels, reloaded on construction.
    call_log_path : path, optional
        JSONL audit log of every request: raw reply, finish reason, token
        usage, served model version and timestamp.
    """

    def __init__(
        self,
        *,
        prompt_template: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        decoding_config: Mapping[str, Any],
        articles: Optional[ArticleSource] = None,
        client: Any = None,
        deployment: Optional[str] = None,
        study_config: Optional[Mapping[str, Any]] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
        reasoning_effort: Optional[str] = "minimal",
        send_sampling_params: bool = False,
        response_format: str = "json_schema",
        max_transport_retries: int = 6,
        backoff_base_seconds: float = 1.0,
        backoff_cap_seconds: float = 60.0,
        per_article_attempts: int = 3,
        max_workers: int = 8,
        label_cache_path: Optional[Union[str, os.PathLike]] = None,
        call_log_path: Optional[Union[str, os.PathLike]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # --- The frozen prompt and its provenance --------------------------
        template = prompt_template.get("template")
        version = prompt_template.get("prompt_version")
        if not isinstance(template, str) or not template:
            raise TeacherConfigurationError("prompt_template['template'] is missing.")
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if version != digest:
            raise TeacherConfigurationError(
                "prompt_template['prompt_version'] is not the SHA-256 of its "
                f"template ({version!r} != {digest!r}); the prompt has drifted."
            )
        attributes = tuple(prompt_template.get("attributes", ATTRIBUTE_NAMES))
        if attributes != ATTRIBUTE_NAMES:
            raise TeacherConfigurationError(
                f"prompt attributes {attributes} differ from the contract {ATTRIBUTE_NAMES}."
            )
        taxonomy = tuple(prompt_template.get("taxonomy_tags") or ())
        if len(taxonomy) != 17 or NONE_LABEL in taxonomy:
            raise TeacherConfigurationError(
                f"prompt_template['taxonomy_tags'] must hold the 17 tags; got {taxonomy}."
            )
        self.system_prompt: str = template
        self.prompt_version: str = digest
        self.taxonomy: Tuple[str, ...] = taxonomy

        # --- Decoding --------------------------------------------------------
        self.model_name: str = str(
            decoding_config.get("teacher_model_name") or DEFAULT_MODEL_NAME
        )
        self.max_output_tokens: int = int(decoding_config["max_output_tokens"])
        window = decoding_config.get("max_context_length_tokens")
        self.max_context_length_tokens: Optional[int] = None if window is None else int(window)
        self.temperature: float = float(decoding_config.get("temperature", 0.0))
        self.top_p: float = float(decoding_config.get("top_p", 1.0))
        self.reasoning_effort = reasoning_effort
        self.send_sampling_params = bool(send_sampling_params)
        if not self.send_sampling_params and (self.temperature, self.top_p) != (1.0, 1.0):
            warnings.warn(
                f"Task 12 pins temperature={self.temperature} and top_p={self.top_p}, "
                f"but {self.model_name} is a reasoning model whose API accepts only the "
                "defaults, so they are not sent; determinism is distributional only. "
                "Pass send_sampling_params=True for a model that accepts them.",
                RuntimeWarning,
                stacklevel=2,
            )
        if response_format == "json_schema":
            self._response_format: Dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "teacher_label",
                    "strict": True,
                    "schema": api_response_schema(response_schema),
                },
            }
        elif response_format == "json_object":
            self._response_format = {"type": "json_object"}
        else:
            raise TeacherConfigurationError(
                f"response_format must be 'json_schema' or 'json_object'; got {response_format!r}."
            )

        # --- Transport ---------------------------------------------------------
        self._client = client if client is not None else build_azure_client(study_config)
        self.deployment: str = deployment or resolve_deployment(study_config)
        self.count_tokens = count_tokens or whitespace_token_count
        self.max_transport_retries = int(max_transport_retries)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.backoff_cap_seconds = float(backoff_cap_seconds)
        self.per_article_attempts = int(per_article_attempts)
        self.max_workers = max(1, int(max_workers))
        self._sleep = sleep

        # --- State ---------------------------------------------------------------
        self._lock = threading.Lock()
        self._article_source = articles
        self._texts: Optional[Dict[str, Tuple[str, str]]] = None
        self._labels: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._label_cache_path = Path(label_cache_path) if label_cache_path else None
        self._call_log_path = Path(call_log_path) if call_log_path else None
        self.stats: Dict[str, Any] = {
            "requests": 0,
            "transport_retries": 0,
            "transport_failures": {},
            "finish_reasons": {},
            "content_filtered": 0,
            "refusals": 0,
            "invalid_replies": {},
            "labels_bought": 0,
            "labels_reused": 0,
            "unknown_article_ids": 0,
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "served_models": {},
        }
        if self._label_cache_path is not None and self._label_cache_path.is_file():
            self._load_label_cache()

    # ------------------------------------------------------------------
    # The three notebook services
    # ------------------------------------------------------------------

    def teacher_client(self, request: Mapping[str, Any]) -> str:
        """Task 13's ``teacher_client(request) -> str``.

        Sends the request Task 13 built (its system prompt and its already
        truncated user prompt) and returns the reply text, which Task 13
        parses, validates and retries itself. Returns "" when the content
        filter blocks the article, so Task 13 quarantines it.
        """
        if not isinstance(request, Mapping):
            raise TypeError("request must be a mapping.")
        self._check_provenance(
            request.get("prompt_version"), request.get("model"), request.get("system_prompt")
        )
        requested_cap = request.get("max_tokens")
        if requested_cap is not None and int(requested_cap) != self.max_output_tokens:
            raise TeacherConfigurationError(
                f"request max_tokens={requested_cap} differs from Task 12's "
                f"pinned {self.max_output_tokens}."
            )
        content, _ = self._complete(
            str(request.get("user_prompt", "")), str(request.get("idempotency_key"))
        )
        return content

    def teacher_call(self, article_id: str, prompt_version: str) -> Optional[Dict[str, Any]]:
        """Tasks 16 and 17's ``teacher_call(article_id, prompt_version) -> Mapping | None``.

        Returns the seven contract fields (``secondary_tag`` None when there is
        no second event) plus the API's ``usage`` mapping, which Task 16 reads
        for its prefix-cache ledger. Returns None for an unusable reply, which
        the notebook counts as a consumed attempt.
        """
        self._check_provenance(prompt_version, None, None)
        key = (self.prompt_version, str(article_id))
        with self._lock:
            cached = self._labels.get(key)
            if cached is not None:
                self.stats["labels_reused"] += 1
        if cached is not None:
            return {name: cached[name] for name in CONTRACT_FIELDS}
        return self._buy_label(str(article_id))

    def teacher_batch_client(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Task 18's ``teacher_client(frame) -> frame``.

        Receives a frame of ``article_id`` and returns one row per requested
        identifier with ``primary_tag``, ``secondary_tag`` ("none" when absent,
        as the stand-in does) and ``labeled_at_utc``. Articles are labelled
        concurrently. If any article still fails after its attempts, a
        RuntimeError is raised, so Task 18 retries the batch; labels already
        bought in it are reused, not re-bought.
        """
        if not isinstance(batch, pd.DataFrame) or "article_id" not in batch.columns:
            raise TypeError("batch must be a DataFrame carrying article_id.")
        identifiers = list(dict.fromkeys(batch["article_id"].astype(str)))
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(identifiers)))) as pool:
            outcomes = list(pool.map(self._label_with_attempts, identifiers))
        failed = [identifier for identifier, label in zip(identifiers, outcomes) if label is None]
        if failed:
            raise RuntimeError(
                f"the teacher could not label {len(failed)} of {len(identifiers)} "
                f"article(s) after {self.per_article_attempts} attempt(s) each; "
                f"first failures: {failed[:5]}."
            )
        rows: List[Dict[str, Any]] = []
        for identifier in identifiers:
            with self._lock:
                record = self._labels[(self.prompt_version, identifier)]
            rows.append(
                {
                    "article_id": identifier,
                    "primary_tag": record["primary_tag"],
                    "secondary_tag": record["secondary_tag"] or NONE_LABEL,
                    "labeled_at_utc": record["labeled_at_utc"],
                }
            )
        return pd.DataFrame(rows, columns=["article_id", "primary_tag", "secondary_tag", "labeled_at_utc"])

    def services(self) -> Dict[str, Callable[..., Any]]:
        """The three entries to merge into the pipeline bundle's ``services``."""
        return {
            "teacher_client": self.teacher_client,
            "teacher_call": self.teacher_call,
            "teacher_batch_client": self.teacher_batch_client,
        }

    def audit(self) -> Dict[str, Any]:
        """A JSON-safe snapshot of the call ledger and the decoding actually used."""
        with self._lock:
            snapshot = json.loads(json.dumps(self.stats))
        snapshot.update(
            {
                "model_name": self.model_name,
                "deployment": self.deployment,
                "prompt_version": self.prompt_version,
                "max_completion_tokens": self.max_output_tokens,
                "reasoning_effort": self.reasoning_effort,
                "sampling_params_sent": self.send_sampling_params,
                "response_format": self._response_format["type"],
            }
        )
        return snapshot

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_provenance(
        self, prompt_version: Any, model: Any, system_prompt: Any
    ) -> None:
        """Refuse a call made under a different prompt, model or system message."""
        if prompt_version != self.prompt_version:
            raise TeacherConfigurationError(
                f"prompt_version {prompt_version!r} differs from the frozen "
                f"{self.prompt_version!r}."
            )
        if model is not None and model != self.model_name:
            raise TeacherConfigurationError(
                f"request model {model!r} differs from the teacher {self.model_name!r}."
            )
        if system_prompt is not None and system_prompt != self.system_prompt:
            raise TeacherConfigurationError(
                "request system_prompt differs from Task 12's frozen template."
            )

    def _resolve_texts(self) -> Dict[str, Tuple[str, str]]:
        """Index the corpus once, by article_id, using Task 13's text columns."""
        with self._lock:
            if self._texts is not None:
                return self._texts
            source = self._article_source
            if source is None:
                raise TeacherConfigurationError(
                    "teacher_call / teacher_batch_client need the article corpus: "
                    "pass articles= (DataFrame, path or callable)."
                )
            if callable(source) and not isinstance(source, pd.DataFrame):
                frame = source()
            elif isinstance(source, pd.DataFrame):
                frame = source
            else:
                path = Path(source)
                frame = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_parquet(path)
            if not isinstance(frame, pd.DataFrame) or "article_id" not in frame.columns:
                raise TeacherConfigurationError("the article corpus must carry article_id.")

            def clean(value: Any) -> str:
                return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)

            texts: Dict[str, Tuple[str, str]] = {}
            if "article_text" in frame.columns:
                # Task 13: a declared text column fills the title slot alone.
                for article_id, text in zip(frame["article_id"], frame["article_text"]):
                    texts[str(article_id)] = (clean(text), "")
            elif {"title", "body"}.issubset(frame.columns):
                for article_id, title, body in zip(frame["article_id"], frame["title"], frame["body"]):
                    texts[str(article_id)] = (clean(title), clean(body))
            else:
                raise TeacherConfigurationError(
                    "the article corpus carries neither article_text nor title and body."
                )
            self._texts = texts
            return texts

    def _user_prompt_for(self, article_id: str) -> Optional[str]:
        """Compose the user message Task 13 would send for this article."""
        texts = self._resolve_texts()
        if article_id not in texts:
            with self._lock:
                self.stats["unknown_article_ids"] += 1
            return None
        title, body = texts[article_id]
        # Task 13's prompt prefix: system prompt, newline, empty user template.
        text, _ = compose_article_suffix(
            title,
            body,
            self.system_prompt + "\n",
            self.max_output_tokens,
            self.max_context_length_tokens,
            self.count_tokens,
        )
        return text

    def _label_with_attempts(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Buy (or reuse) one label with up to per_article_attempts tries."""
        for _ in range(self.per_article_attempts):
            try:
                label = self.teacher_call(article_id, self.prompt_version)
            except (ConnectionError, TimeoutError, RuntimeError):
                continue
            if label is not None:
                return label
        return None

    def _buy_label(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Send one article to the teacher, validate the reply and cache the label."""
        user_prompt = self._user_prompt_for(article_id)
        if user_prompt is None:
            return None
        content, meta = self._complete(user_prompt, article_id)
        label, reason = validate_label(content, self.taxonomy)
        if label is None:
            key = reason.split(" ")[0] if reason else "empty"
            if meta.get("content_filtered"):
                key = "content_filter"
            elif content == "":
                key = f"empty({meta.get('finish_reason')})"
            with self._lock:
                self.stats["invalid_replies"][key] = self.stats["invalid_replies"].get(key, 0) + 1
            return None
        record = dict(label)
        record["labeled_at_utc"] = meta["labeled_at_utc"]
        record["served_model"] = meta.get("model")
        with self._lock:
            self._labels[(self.prompt_version, article_id)] = record
            self.stats["labels_bought"] += 1
            if self._label_cache_path is not None:
                self._append_jsonl(
                    self._label_cache_path,
                    {"prompt_version": self.prompt_version, "article_id": article_id, **record},
                )
        result = {name: label[name] for name in CONTRACT_FIELDS}
        if meta.get("usage"):
            result["usage"] = meta["usage"]
        return result

    def _complete(self, user_prompt: str, article_id: str) -> Tuple[str, Dict[str, Any]]:
        """One logical request with transport retries; returns (content, meta).

        Persistent transport failure is raised as a built-in exception class
        the notebook's retry loops catch. A prompt blocked by Azure's content
        filter returns "" with meta["content_filtered"] set.
        """
        kwargs: Dict[str, Any] = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "response_format": self._response_format,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.send_sampling_params:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p

        last_error: Optional[BaseException] = None
        for retry in range(self.max_transport_retries + 1):
            with self._lock:
                self.stats["requests"] += 1
            try:
                response = self._client.chat.completions.create(**kwargs)
            except openai.BadRequestError as error:
                if _is_content_filter(error):
                    with self._lock:
                        self.stats["content_filtered"] += 1
                    meta = {"content_filtered": True, "finish_reason": "content_filter",
                            "labeled_at_utc": _now_utc()}
                    self._log_call(article_id, "", meta, error=repr(error))
                    return "", meta
                raise  # a malformed request is a configuration bug: halt
            except (openai.APIConnectionError, openai.APIStatusError) as error:
                status = getattr(error, "status_code", None)
                if isinstance(error, openai.APIStatusError) and status not in RETRYABLE_STATUS:
                    raise  # 401 / 403 / 404 ...: halt, do not burn retries
                last_error = error
                with self._lock:
                    name = type(error).__name__
                    self.stats["transport_failures"][name] = self.stats["transport_failures"].get(name, 0) + 1
                if retry < self.max_transport_retries:
                    with self._lock:
                        self.stats["transport_retries"] += 1
                    self._sleep(self._backoff_seconds(retry, error))
                    continue
                break
            return self._read_response(response, article_id)

        self._log_call(article_id, None, {"labeled_at_utc": _now_utc()}, error=repr(last_error))
        message = (
            f"teacher request for article {article_id!r} failed after "
            f"{self.max_transport_retries + 1} attempt(s): {last_error!r}"
        )
        if isinstance(last_error, openai.APITimeoutError):
            raise TimeoutError(message) from last_error
        if isinstance(last_error, openai.APIConnectionError):
            raise ConnectionError(message) from last_error
        raise RuntimeError(message) from last_error

    def _read_response(self, response: Any, article_id: str) -> Tuple[str, Dict[str, Any]]:
        """Extract the reply text and the ledger metadata from one completion."""
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        refusal = getattr(message, "refusal", None)
        finish_reason = getattr(choice, "finish_reason", None)
        usage = _usage_dict(getattr(response, "usage", None))
        meta: Dict[str, Any] = {
            "finish_reason": finish_reason,
            "model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "usage": usage,
            "content_filtered": finish_reason == "content_filter",
            "labeled_at_utc": _now_utc(),
        }
        with self._lock:
            stats = self.stats
            stats["finish_reasons"][str(finish_reason)] = stats["finish_reasons"].get(str(finish_reason), 0) + 1
            if meta["content_filtered"]:
                stats["content_filtered"] += 1
            if refusal:
                stats["refusals"] += 1
            if meta["model"]:
                stats["served_models"][meta["model"]] = stats["served_models"].get(meta["model"], 0) + 1
            if usage:
                stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                stats["cached_prompt_tokens"] += int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                )
                stats["reasoning_tokens"] += int(
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                )
        if finish_reason == "length" and not content:
            warnings.warn(
                f"article {article_id!r}: the {self.max_output_tokens}-token cap was spent "
                "before any output (reasoning tokens count against it); lower "
                "reasoning_effort or raise the cap.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._log_call(article_id, content, meta, refusal=refusal)
        return content, meta

    def _backoff_seconds(self, retry: int, error: BaseException) -> float:
        """Honour Retry-After when the service sends it, else capped exponential backoff with jitter."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or {}
        for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            value = headers.get(header) if hasattr(headers, "get") else None
            if value:
                try:
                    return min(self.backoff_cap_seconds, max(0.0, float(value) * scale))
                except ValueError:
                    pass
        delay = self.backoff_base_seconds * (2 ** retry)
        return min(self.backoff_cap_seconds, delay) * (0.5 + random.random() / 2)

    def _log_call(self, article_id: str, content: Optional[str], meta: Mapping[str, Any], **extra: Any) -> None:
        """Append one request to the audit log, when one is configured."""
        if self._call_log_path is None:
            return
        entry = {
            "article_id": article_id,
            "prompt_version": self.prompt_version,
            "deployment": self.deployment,
            "raw_response_text": content,
            **{key: value for key, value in meta.items()},
            **extra,
        }
        with self._lock:
            self._append_jsonl(self._call_log_path, entry)

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        """Append one JSON line (caller holds the lock)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _load_label_cache(self) -> None:
        """Reload labels bought by an earlier run under the same prompt version."""
        assert self._label_cache_path is not None
        with self._label_cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("prompt_version") != self.prompt_version:
                    continue
                label, reason = validate_label(
                    {name: record.get(name) for name in CONTRACT_FIELDS}, self.taxonomy
                )
                if label is None:
                    continue
                label["labeled_at_utc"] = record.get("labeled_at_utc")
                label["served_model"] = record.get("served_model")
                self._labels[(self.prompt_version, str(record["article_id"]))] = label


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _now_utc() -> str:
    """The current UTC instant in Task 18's isoformat."""
    return pd.Timestamp.now(tz="UTC").isoformat()


def _usage_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """Render the SDK's usage object as a plain JSON-safe mapping."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    if not isinstance(usage, Mapping):
        return None
    return json.loads(json.dumps(usage, default=str))


def _is_content_filter(error: BaseException) -> bool:
    """Whether an Azure 400 is a prompt blocked by the content filter."""
    if getattr(error, "code", None) == "content_filter":
        return True
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        inner = body.get("error", body)
        if isinstance(inner, Mapping) and inner.get("code") == "content_filter":
            return True
        if body.get("code") == "content_filter":
            return True
    return False


def make_azure_teacher_services(**kwargs: Any) -> Tuple[AzureTeacher, Dict[str, Callable[..., Any]]]:
    """Build an :class:`AzureTeacher` and return it with its three services."""
    teacher = AzureTeacher(**kwargs)
    return teacher, teacher.services()


__all__ = [
    "ATTRIBUTE_NAMES",
    "CONTRACT_FIELDS",
    "AzureTeacher",
    "TeacherConfigurationError",
    "api_response_schema",
    "build_azure_client",
    "compose_article_suffix",
    "make_azure_teacher_services",
    "resolve_deployment",
    "tiktoken_counter",
    "validate_label",
    "whitespace_token_count",
]
