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
    """
    Raised when a provenance or configuration mismatch must halt the run.

        A call made under a different prompt version, model or system prompt,
        or a template whose digest does not match its declared version, would
        relabel articles under a drifted prompt and break the provenance every
        teacher label records. The exception subclasses ``ValueError``, which
        none of the notebook's retry loops catch, so the stage fails instead of
        silently spending calls on the wrong contract.
    """


# ----------------------------------------------------------------------------
# Token counting and truncation (mirrors Task 13's _compose_teacher_suffix)
# ----------------------------------------------------------------------------


def whitespace_token_count(text: str) -> int:
    """
    Count the whitespace-separated tokens of a text.

        This is the notebook's default counter (Tasks 12 and 13), which the
        pipeline uses because stage 13 receives no ``count_tokens``. Using it
        here keeps the ID-only services' truncation identical to Task 13's.

        Parameters
        ----------
        text : str
            The text to count.

        Returns
        -------
        int
            The number of whitespace-separated tokens.

        Examples
        --------
        >>> whitespace_token_count("Revenue rose 12% on earnings.")
        5
    """
    return len(text.split())


def tiktoken_counter(encoding_name: str = "o200k_base") -> Callable[[str], int]:
    """
    Build a BPE token counter for the gpt-5 family's tokenizer.

        The counter is exact where the whitespace default is an approximation,
        but it changes which characters survive truncation. Use it only if
        Task 13 receives the same counter through ``count_tokens``; otherwise
        Tasks 13 and 16-18 would send the teacher differently truncated text.

        Parameters
        ----------
        encoding_name : str, optional
            The ``tiktoken`` encoding name. Defaults to ``"o200k_base"``.

        Returns
        -------
        Callable[[str], int]
            A function returning the token count of its argument.

        Raises
        ------
        ImportError
            If the optional ``tiktoken`` package is not installed.
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
    """
    Join and truncate the article exactly as Task 13 does.

        Re-implements ``_compose_teacher_suffix_task_13`` so that the ID-only
        services send the teacher the same text Task 13 would. The candidate
        suffix is ``title + "\\n\\n" + body``, except that a part which is empty
        after stripping is dropped together with the separator. The remaining
        budget is the context window less the output allowance and the token
        count of ``prompt_prefix``; the longest character prefix of the
        candidate whose token count fits that budget is returned. An article
        with no text at all yields the empty suffix, which Task 13 also sends.

        Parameters
        ----------
        title : str
            The article's title.
        body : str
            The article's body.
        prompt_prefix : str
            The cached prompt text that precedes the suffix.
        max_output_tokens : int
            The output allowance (Task 12's ``max_output_tokens``).
        max_context_length_tokens : Optional[int]
            The context window (DEF15); ``None`` disables truncation.
        count_tokens : Callable[[str], int], optional
            The token counter. Defaults to :func:`whitespace_token_count`.

        Returns
        -------
        Tuple[str, bool]
            ``(text, truncated)`` where ``truncated`` is ``True`` exactly when the
            returned text is shorter than the candidate suffix.

        Raises
        ------
        ValueError
            If the remaining budget is non-positive.

        Examples
        --------
        >>> compose_article_suffix("Markets", "", "prefix", 16, None)
        ('Markets', False)
        >>> compose_article_suffix("Markets", "Stocks rose.", "prefix", 2, 5)[1]
        True
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
    """
    Reduce Task 12's schema to the subset Azure strict structured outputs accept.

        Strict mode supports ``type``, ``properties``, ``enum``, ``required`` and
        ``additionalProperties`` but not the ``allOf`` if/then branches Task 12
        uses to forbid ``secondary_tag == primary_tag``, nor the ``x_*``
        metadata block. The seven properties, their enums and the closed object
        are kept verbatim; the one dropped rule is enforced locally by
        :func:`validate_label`.

        Parameters
        ----------
        task12_schema : Mapping[str, Any]
            The return value of ``build_teacher_response_schema_task_12``.

        Returns
        -------
        Dict[str, Any]
            A closed JSON Schema over exactly the seven contract fields.

        Raises
        ------
        TeacherConfigurationError
            If the schema does not declare exactly the seven contract fields.
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
    """
    Check one teacher reply against the frozen seven-field contract.

        The reply must be a JSON object carrying exactly the seven fields; the
        primary tag must be one of the declared tags; the secondary tag must be
        one of them or the ``"none"`` literal, and may not repeat the primary
        tag; and the five attributes must be booleans. Task 17 halts on an
        off-taxonomy tag, so the check runs before any service returns a label.

        Parameters
        ----------
        reply : Any
            The reply, as a JSON string or an already decoded mapping.
        taxonomy : Tuple[str, ...]
            The seventeen admissible tag names.

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Optional[str]]
            ``(label, None)`` with ``secondary_tag`` normalised to ``None`` when
            the teacher reports no second event, or ``(None, reason)`` when the
            reply is unusable.

        Examples
        --------
        >>> validate_label(
        ...     '{"primary_tag": "M&A deal", "secondary_tag": "none", '
        ...     '"scheduled": false, "forward_looking": true, '
        ...     '"primary_source": false, "quantified": false, "rumor": true}',
        ...     ("M&A deal", "Earnings results"),
        ... )[0]["secondary_tag"] is None
        True
        >>> validate_label('{"primary_tag": "Weather"}', ("M&A deal",))[1]
        "missing fields ['secondary_tag', 'scheduled', 'forward_looking', \
'primary_source', 'quantified', 'rumor']"
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
    """
    Read the teacher's environment-variable names from the study configuration.

        Parameters
        ----------
        study_config : Optional[Mapping[str, Any]]
            The study configuration, whose ``llm_api.teacher`` block names the
            variables, or ``None``.

        Returns
        -------
        Dict[str, str]
            The four variable names, falling back to :data:`DEFAULT_ENV_VARS`
            for any the configuration leaves unset.
    """
    names = dict(DEFAULT_ENV_VARS)
    block = ((study_config or {}).get("llm_api") or {}).get("teacher") or {}
    for key in names:
        if block.get(key):
            names[key] = str(block[key])
    return names


def resolve_deployment(study_config: Optional[Mapping[str, Any]] = None) -> str:
    """
    Return the Azure deployment name from its environment variable.

        Parameters
        ----------
        study_config : Optional[Mapping[str, Any]], optional
            The study configuration naming the variable. Defaults to ``None``,
            which reads ``AZURE_OPENAI_DEPLOYMENT``.

        Returns
        -------
        str
            The deployment name.

        Raises
        ------
        TeacherConfigurationError
            If the variable is unset or empty.
    """
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
    """
    Build an Azure OpenAI client from the environment variables config.yaml names.

        The key is read from the environment and never stored in the
        configuration, which keeps the study's ``api_key: null``. The SDK's own
        retries are switched off (``max_retries=0``) so that
        :class:`AzureTeacher` sees and counts every attempt.

        Parameters
        ----------
        study_config : Optional[Mapping[str, Any]], optional
            The study configuration naming the variables. Defaults to ``None``,
            which reads the :data:`DEFAULT_ENV_VARS` names.
        timeout : float, optional
            The per-request timeout in seconds. Keyword-only; defaults to 60.0.

        Returns
        -------
        openai.AzureOpenAI
            The configured client.

        Raises
        ------
        TeacherConfigurationError
            If the key, endpoint or API-version variable is unset or empty.
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
    """
    The study's gpt-5-mini teacher on Azure OpenAI, behind the notebook's three service shapes.

        One instance holds the client, the frozen Task 12 prompt and the article
        corpus, and exposes ``teacher_client`` (Task 13), ``teacher_call``
        (Tasks 16 and 17) and ``teacher_batch_client`` (Task 18). Every argument
        is keyword-only.

        Parameters
        ----------
        prompt_template : Mapping[str, Any]
            The return value of ``build_teacher_prompt_template_task_12``. Task
            12's builders are pure functions of the configuration, so this equals
            what stage 12 produces, including the ``prompt_version`` digest.
        response_schema : Mapping[str, Any]
            The return value of ``build_teacher_response_schema_task_12``.
        decoding_config : Mapping[str, Any]
            The return value of ``build_teacher_decoding_config_task_12``.
        articles : Optional[ArticleSource], optional
            The corpus the ID-only services resolve text from: the cleansed
            ``retained_articles_c`` that Task 13 labels, as a DataFrame, a
            Parquet or pickle path, or a zero-argument callable. A path or
            callable is read lazily on the first ID-only call, so it can point at
            ``<output_dir>/stage_outputs/retained_articles_c.parquet``, which
            stage 3 writes before any teacher stage runs. Defaults to ``None``,
            which leaves only ``teacher_client`` usable.
        client : Any, optional
            An ``openai.AzureOpenAI`` or compatible client. Defaults to ``None``,
            which builds one with :func:`build_azure_client`.
        deployment : Optional[str], optional
            The Azure deployment name. Defaults to ``None``, which reads
            ``$AZURE_OPENAI_DEPLOYMENT``.
        study_config : Optional[Mapping[str, Any]], optional
            The study configuration naming the environment variables. Defaults
            to ``None``.
        count_tokens : Optional[Callable[[str], int]], optional
            The truncation counter. It must be the one Task 13 uses; the pipeline
            passes none, so defaults to ``None``, the whitespace counter.
        reasoning_effort : Optional[str], optional
            Sent to the API; ``None`` omits it. Defaults to ``"minimal"``, which
            keeps hidden reasoning inside the pinned 200-token cap.
        send_sampling_params : bool, optional
            Whether to send the pinned temperature and top_p. Defaults to
            ``False`` because gpt-5 models reject them.
        response_format : str, optional
            ``"json_schema"`` for strict structured outputs, per
            ``teacher.output_format``, or ``"json_object"`` for plain JSON mode.
            Defaults to ``"json_schema"``.
        max_transport_retries : int, optional
            Retries of one request on a 429, a 5xx or a timeout before giving up.
            Defaults to 6.
        backoff_base_seconds : float, optional
            The first exponential-backoff delay. Defaults to 1.0.
        backoff_cap_seconds : float, optional
            The longest backoff delay, ``Retry-After`` included. Defaults to 60.0.
        per_article_attempts : int, optional
            Logical attempts per article inside ``teacher_batch_client``.
            Defaults to 3.
        max_workers : int, optional
            Concurrent requests inside ``teacher_batch_client``. Defaults to 8.
        label_cache_path : Optional[Union[str, os.PathLike]], optional
            A JSONL file of purchased labels, reloaded on construction so a rerun
            never re-buys a label. Defaults to ``None``, an in-memory cache only.
        call_log_path : Optional[Union[str, os.PathLike]], optional
            A JSONL audit log of every request: raw reply, finish reason, token
            usage, served model version and timestamp. Defaults to ``None``.
        sleep : Callable[[float], None], optional
            The backoff sleep, injectable for tests. Defaults to ``time.sleep``.

        Attributes
        ----------
        system_prompt : str
            Task 12's frozen template, sent as the system message.
        prompt_version : str
            The template's SHA-256 digest.
        taxonomy : Tuple[str, ...]
            The seventeen admissible tag names.
        model_name : str
            The teacher model the prompt was frozen against.
        deployment : str
            The Azure deployment every request is sent to.
        stats : Dict[str, Any]
            The running call ledger; read it through :meth:`audit`.

        Raises
        ------
        TeacherConfigurationError
            If the template's digest differs from its ``prompt_version``, the
            attributes or taxonomy differ from the contract, the response format
            is unknown, the schema lacks a contract field, or a required
            environment variable is unset.

        Warns
        -----
        RuntimeWarning
            When the pinned temperature and top_p are not sent.
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
        """
        Answer one Task 13 request (service ``teacher_client``).

                Sends the request Task 13 built, its system prompt and its already
                truncated user prompt, and returns the reply text, which Task 13
                parses, validates and retries itself. The request's prompt version,
                model, system prompt and output cap must match the frozen ones.

                Parameters
                ----------
                request : Mapping[str, Any]
                    The request built by Task 13's ``_teacher_request``.

                Returns
                -------
                str
                    The reply text, or ``""`` when the content filter blocks the
                    article, so Task 13 quarantines it.

                Raises
                ------
                TypeError
                    If ``request`` is not a mapping.
                TeacherConfigurationError
                    If the request drifts from the frozen prompt, model or cap.
                ConnectionError, TimeoutError, RuntimeError
                    If a transport failure persists, so Task 13 retries the request.
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
        """
        Buy one label by article identifier (service ``teacher_call``).

                Serves Tasks 16 and 17. The article's text is resolved from the
                corpus and truncated by Task 13's rule, so the request equals the
                one Task 13 would send. A label already bought under this prompt
                version is returned without a request.

                Parameters
                ----------
                article_id : str
                    The article's identifier, the study's idempotency key.
                prompt_version : str
                    The frozen prompt's version, which must equal this teacher's.

                Returns
                -------
                Optional[Dict[str, Any]]
                    The seven contract fields, with ``secondary_tag`` ``None`` when
                    there is no second event, plus the API's ``usage`` mapping on a
                    fresh purchase, which Task 16 reads for its prefix-cache ledger;
                    or ``None`` for an unusable reply or an unknown identifier, which
                    the notebook counts as a consumed attempt.

                Raises
                ------
                TeacherConfigurationError
                    If ``prompt_version`` differs from the frozen version.
                ConnectionError, TimeoutError, RuntimeError
                    If a transport failure persists, so the caller retries.
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
        """
        Label one routed batch (service ``teacher_batch_client``).

                Serves Task 18's ``teacher_client`` binding. The articles are
                labelled concurrently, each with up to ``per_article_attempts``
                tries. If any article still fails, the whole batch raises, so Task
                18 retries it; labels already bought in the batch are reused on the
                retry, never re-bought.

                Parameters
                ----------
                batch : pd.DataFrame
                    The routed rows, carrying ``article_id``.

                Returns
                -------
                pd.DataFrame
                    One row per requested identifier with ``article_id``,
                    ``primary_tag``, ``secondary_tag`` (the ``"none"`` literal when
                    absent, as the stand-in returns) and ``labeled_at_utc``.

                Raises
                ------
                TypeError
                    If ``batch`` is not a DataFrame carrying ``article_id``.
                RuntimeError
                    If any article in the batch could not be labelled.
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
        """
        Return the three entries to merge into the pipeline bundle's services.

                Returns
                -------
                Dict[str, Callable[..., Any]]
                    ``teacher_client``, ``teacher_call`` and ``teacher_batch_client``,
                    keyed by the service names Task 29's stage catalog binds.
        """
        return {
            "teacher_client": self.teacher_client,
            "teacher_call": self.teacher_call,
            "teacher_batch_client": self.teacher_batch_client,
        }

    def audit(self) -> Dict[str, Any]:
        """
        Snapshot the call ledger and the decoding actually used.

                Returns
                -------
                Dict[str, Any]
                    A JSON-safe copy of :attr:`stats` (requests, retries, failures by
                    class, finish reasons, invalid replies, labels bought and reused,
                    token totals and served model versions), extended with the model,
                    deployment, prompt version and decoding parameters sent.
        """
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
        """
        Refuse a call made under a different prompt, model or system message.

                Parameters
                ----------
                prompt_version : Any
                    The caller's prompt version; it must equal the frozen digest.
                model : Any
                    The caller's model name, or ``None`` to skip the check.
                system_prompt : Any
                    The caller's system prompt, or ``None`` to skip the check.

                Raises
                ------
                TeacherConfigurationError
                    If any supplied value differs from the frozen one.
        """
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
        """
        Index the corpus once by article identifier, using Task 13's text columns.

                A declared ``article_text`` column fills the title slot alone, as in
                Task 13; otherwise ``title`` and ``body`` are read. Missing values
                become empty strings.

                Returns
                -------
                Dict[str, Tuple[str, str]]
                    ``article_id -> (title, body)``.

                Raises
                ------
                TeacherConfigurationError
                    If no corpus was supplied, or it lacks ``article_id`` or both
                    text layouts.
        """
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
                """
                Render one text cell as a string, a missing value as empty.

                        Parameters
                        ----------
                        value : Any
                            The cell's value.

                        Returns
                        -------
                        str
                            The text, or ``""`` for ``None`` or a missing value.
                """
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
        """
        Compose the user message Task 13 would send for one article.

                Parameters
                ----------
                article_id : str
                    The article's identifier.

                Returns
                -------
                Optional[str]
                    The truncated article suffix, or ``None`` when the corpus does not
                    hold the identifier.
        """
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
        """
        Buy or reuse one label with up to ``per_article_attempts`` tries.

                Parameters
                ----------
                article_id : str
                    The article's identifier.

                Returns
                -------
                Optional[Dict[str, Any]]
                    The label, or ``None`` when every attempt failed.
        """
        for _ in range(self.per_article_attempts):
            try:
                label = self.teacher_call(article_id, self.prompt_version)
            except (ConnectionError, TimeoutError, RuntimeError):
                continue
            if label is not None:
                return label
        return None

    def _buy_label(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Send one article to the teacher, validate the reply and cache the label.

                The label is cached in memory and, when configured, appended to the
                label cache file with its timestamp and the served model version.
                An unusable reply is counted in ``stats["invalid_replies"]`` by
                reason.

                Parameters
                ----------
                article_id : str
                    The article's identifier.

                Returns
                -------
                Optional[Dict[str, Any]]
                    The seven contract fields plus ``usage`` when reported, or
                    ``None`` for an unknown identifier or an unusable reply.
        """
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
        """
        Send one logical request, retrying transport failures with backoff.

                A 408, 409, 429 or 5xx status, a timeout or a dropped connection is
                retried up to ``max_transport_retries`` times. Any other status
                propagates unchanged, because a bad key, an unknown deployment or a
                malformed request is a configuration bug that must halt the run. A
                prompt blocked by Azure's content filter is not an error: it returns
                an empty reply.

                Parameters
                ----------
                user_prompt : str
                    The user message: the truncated article suffix.
                article_id : str
                    The article's identifier, recorded in the call log.

                Returns
                -------
                Tuple[str, Dict[str, Any]]
                    The reply text and its metadata: finish reason, served model,
                    system fingerprint, token usage, the content-filter flag and the
                    UTC timestamp.

                Raises
                ------
                TimeoutError
                    If the last failure was a timeout.
                ConnectionError
                    If the last failure was a dropped connection.
                RuntimeError
                    If the last failure was a retryable status.
                openai.APIStatusError
                    For a non-retryable status, such as 401, 403 or 404.
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
        """
        Extract the reply text and the ledger metadata from one completion.

                Parameters
                ----------
                response : Any
                    The SDK's chat completion.
                article_id : str
                    The article's identifier, recorded in the call log.

                Returns
                -------
                Tuple[str, Dict[str, Any]]
                    The reply text (``""`` when absent) and its metadata.

                Warns
                -----
                RuntimeWarning
                    When the output cap was spent before any reply text, which means
                    hidden reasoning consumed the budget.
        """
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
        """
        Compute the delay before the next transport retry.

                ``Retry-After`` (milliseconds or seconds) is honoured when the service
                sends it; otherwise the delay is capped exponential backoff with
                jitter.

                Parameters
                ----------
                retry : int
                    The zero-based retry number.
                error : BaseException
                    The failure that forced the retry.

                Returns
                -------
                float
                    The delay in seconds, at most ``backoff_cap_seconds``.
        """
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
        """
        Append one request to the audit log, when one is configured.

                Parameters
                ----------
                article_id : str
                    The article's identifier.
                content : Optional[str]
                    The raw reply text, or ``None`` when no reply arrived.
                meta : Mapping[str, Any]
                    The reply's metadata.
                **extra : Any
                    Further fields to record, such as the error or the refusal.
        """
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
        """
        Append one JSON line to a file; the caller holds the lock.

                Parameters
                ----------
                path : Path
                    The JSONL file, created with its parents when absent.
                record : Mapping[str, Any]
                    The record to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _load_label_cache(self) -> None:
        """
        Reload the labels an earlier run bought under the same prompt version.

                Records under another prompt version, and records that fail the
                label contract, are skipped rather than trusted.
        """
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
    """
    Return the current UTC instant in Task 18's timestamp format.

        Returns
        -------
        str
            ``pd.Timestamp.now(tz="UTC").isoformat()``.
    """
    return pd.Timestamp.now(tz="UTC").isoformat()


def _usage_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """
    Render the SDK's usage object as a plain JSON-safe mapping.

        Parameters
        ----------
        usage : Any
            The completion's ``usage`` attribute: a pydantic model, a mapping or
            ``None``.

        Returns
        -------
        Optional[Dict[str, Any]]
            The token counts, nested in OpenAI's shape, or ``None`` when absent.

        Examples
        --------
        >>> _usage_dict({"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 0}})
        {'prompt_tokens': 900, 'prompt_tokens_details': {'cached_tokens': 0}}
    """
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    if not isinstance(usage, Mapping):
        return None
    return json.loads(json.dumps(usage, default=str))


def _is_content_filter(error: BaseException) -> bool:
    """
    Decide whether an Azure HTTP 400 is a prompt blocked by the content filter.

        Parameters
        ----------
        error : BaseException
            The SDK's ``BadRequestError``.

        Returns
        -------
        bool
            ``True`` when the error, or its body, carries the ``content_filter``
            code.
    """
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
    """
    Build an :class:`AzureTeacher` and return it with its three services.

        Parameters
        ----------
        **kwargs : Any
            The keyword arguments of :class:`AzureTeacher`.

        Returns
        -------
        Tuple[AzureTeacher, Dict[str, Callable[..., Any]]]
            The teacher, for its :meth:`AzureTeacher.audit`, and the services to
            merge into the pipeline bundle.
    """
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
