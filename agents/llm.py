"""JSON client, offline cache, prompt boundary and per-attempt audit log."""
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from config import LLM_MAX_RETRIES, LLM_TIMEOUT_S, SPLITS

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "agents/cache"
LOG_PATH = ROOT / "runs/llm_calls.jsonl"
log = logging.getLogger(__name__)
load_dotenv(ROOT / ".env")

DISC_FIELDS = {
    "n_obs", "sharpe_net_ann", "nw_t", "p_block", "best_month", "worst_month",
    "share_of_return_from_top_month",
}
EXPLAIN_FIELDS = {
    "label", "net_sharpe", "sharpe_net_ann", "sharpe", "p", "p_block",
    "mde_sharpe_80", "mintrl_years", "years", "power_pct", "power_at_05",
    "dsr", "n_obs", "nw_t", "q_by", "p_search", "rule", "as_of", "est",
    "lo", "hi", "cusum", "h", "alarm", "integrity", "months_since_calib_end",
}
SPEC_FIELDS = {"feature", "series", "series2", "lookback", "transform", "asset", "direction"}
REVIEW_FIELDS = {
    "id", "round", "max_rounds", "ideas_tried", "effective_trials", "hurdle_t", "observed_t",
    "p_search", "q_by", "p_block", "nw_t", "sharpe_net_ann", "n_obs", "status", "failed",
    "dup_of", "refine_budget", "nominate_max", "best_month", "worst_month",
    "share_of_return_from_top_month",
}
NUM = re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
YEAR = re.compile(r"(?<!\w)(?:19|20)\d{2}(?!\w)")
MONTH = re.compile(r"(?<!\d)(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?(?!\d)")
HOLDOUT_YEAR = int(SPLITS["holdout"][0][:4])
TRUSTED_NUMBERS = {
    "lookbacks": re.compile(r"^  lookback: one of \[1,3,6,12,24,36\] \(use 1 for \"level\"\)$"),
    "direction": re.compile(r'^  direction: 1 if a HIGH feature value means go LONG the asset next month, -1 if it means go SHORT$'),
}


class LLMError(RuntimeError):
    pass


class OfflineMiss(LLMError):
    pass


class HoldoutViolation(LLMError):
    pass


def offline():
    return os.getenv("OFFLINE", "0") == "1"


def now():
    return datetime.now(timezone.utc)


def _date_safe(text):
    if any(int(year) >= HOLDOUT_YEAR for year in YEAR.findall(text)):
        raise HoldoutViolation("Prompt includes a date or year from the sealed period.")


def assert_no_holdout(text: str):
    """Reject dated holdout material and data payloads outside typed prompt fields."""
    _date_safe(text)
    for line in text.splitlines():
        for prefix, fields in (
            ("Hypothesis spec: ", SPEC_FIELDS),
            ("Discovery summary: ", DISC_FIELDS),
            ("FACTS: ", EXPLAIN_FIELDS),
            ("Engine evidence: ", REVIEW_FIELDS),
        ):
            if not line.startswith(prefix):
                continue
            try:
                obj = json.loads(line[len(prefix):])
            except ValueError:
                raise HoldoutViolation("Prompt contains an invalid evidence payload.") from None
            if not isinstance(obj, dict) or obj.keys() - fields:
                raise HoldoutViolation("Prompt contains an unapproved evidence field.")
            for key, value in obj.items():
                if value is None or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    if prefix != "Hypothesis spec: " or key in {"lookback", "direction"}:
                        continue
                if isinstance(value, str) and _safe_field_text(prefix, key, value):
                    continue
                raise HoldoutViolation(f"Prompt contains unsupported evidence in {key}.")
            break
        else:
            cleaned = YEAR.sub("", MONTH.sub("", line))
            if any(pattern.fullmatch(line) for pattern in TRUSTED_NUMBERS.values()):
                continue
            if line.startswith("  series: one of "):
                cleaned = re.sub(r"\b(?:[1-9]\d?y|[1-9]\d?m|S&P 500)\b", "", cleaned)
            cleaned = re.sub(r"\bseries2\b|L>=12|\bat most 3\b", "", cleaned)
            if re.fullmatch(r"Return (?:exactly|only) \d{1,2} (?:new )?hypotheses\..*", cleaned):
                cleaned = re.sub(r"^Return (?:exactly|only) \d{1,2} ", "", cleaned)
            if NUM.search(cleaned):
                raise HoldoutViolation("Prompt includes numeric data outside approved fields.")


def _safe_field_text(prefix, key, value):
    if prefix == "Hypothesis spec: ":
        return key not in {"lookback", "direction"} and bool(re.fullmatch(r"[A-Za-z][\w]*", value))
    if key in {"best_month", "worst_month", "as_of"}:
        return bool(MONTH.fullmatch(value))
    if prefix == "Engine evidence: ":
        return bool(re.fullmatch(r"[A-Za-z][\w,]*", value))
    if key in {"label", "rule", "integrity"}:
        return not NUM.search(value)
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%?", value))


def prose(text: str) -> str:
    """Free text for a prompt line. Tokens containing digits are dropped because the boundary
    rejects numbers outside the typed evidence fields."""
    return re.sub(r"\s{2,}", " ", re.sub(r"\S*\d\S*", "", text)).strip()


def key_for(model, system, user, temperature):
    return hashlib.sha256((model + system + user + str(temperature)).encode()).hexdigest()


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise OSError(f"Short append to {path.name}")
    finally:
        os.close(fd)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            json.dump(value, tmp, ensure_ascii=False, allow_nan=False)
            tmp.close()
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


def _cache_get(key, schema):
    try:
        cache_dir = Path(os.getenv("LLM_CACHE_DIR") or CACHE_DIR)
        data = json.loads((cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        response = data["response"]
        return schema.model_validate_json(response if isinstance(response, str) else json.dumps(response))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _audit(role, model, key, ok, start, source):
    append_jsonl(LOG_PATH, {
        "role": role, "model": model, "key": key, "ok": ok,
        "latency_s": round(time.monotonic() - start, 4), "ts": now().isoformat(),
        "source": source,
    })


def chat_json(role: str, system: str, user: str, schema: type[BaseModel],
              temperature: float) -> BaseModel:
    primary = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    fallback = os.getenv("LLM_MODEL_FALLBACK") or "gpt-4o"
    models = [(primary, LLM_MAX_RETRIES + 1)]
    if fallback != primary:
        models.append((fallback, 1))
    started = time.monotonic()
    if role != "probe":
        try:
            assert_no_holdout(system)
            assert_no_holdout(user)
        except HoldoutViolation:
            _audit(role, primary, key_for(primary, system, user, temperature), False, started, "blocked")
            raise

    is_offline = offline()
    use_cache = is_offline or os.getenv("USE_CACHE", "1") == "1"
    if is_offline:
        for model in dict.fromkeys(model for model, _ in models):
            key = key_for(model, system, user, temperature)
            cached = _cache_get(key, schema)
            if cached is not None:
                _audit(role, model, key, True, started, "cache")
                return cached
        _audit(role, primary, key_for(primary, system, user, temperature), False, started, "offline_miss")
        raise OfflineMiss(f"{role}: not cached")

    api_key = os.getenv("LLM_API_KEY")
    client = None
    last_error = None
    for model, attempts in models:
        key = key_for(model, system, user, temperature)
        if use_cache and (cached := _cache_get(key, schema)) is not None:
            _audit(role, model, key, True, started, "cache")
            return cached
        if not api_key:
            continue
        if client is None:
            client = OpenAI(
                api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None,
                timeout=LLM_TIMEOUT_S, max_retries=0,
            )
        repair = ""
        rate_limited = False
        for _ in range(attempts):
            start = time.monotonic()
            ok = False
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + repair},
                ]
                result = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    response_format={"type": "json_object"},
                )
                choices = result.choices
                response = choices[0].message.content if choices else None
                parsed = schema.model_validate_json(response or "")
                cache_dir = Path(os.getenv("LLM_CACHE_DIR") or CACHE_DIR)
                write_json(cache_dir / f"{key}.json", {
                    "request": {"system": system, "user": user, "temperature": temperature},
                    "response": response, "model": model, "ts": now().isoformat(),
                })
                ok = True
                return parsed
            except ValidationError as exc:
                last_error = exc
                # Include schema failure locations/types, never echo model output.
                errors = [item["type"] for item in exc.errors(include_input=False)]
                repair = (
                    "\nYour last output failed validation: "
                    + ", ".join(dict.fromkeys(errors)) + ". Return only valid JSON."
                )
            except OpenAIError as exc:
                last_error = exc
                log.info("%s provider attempt failed: %s", role, type(exc).__name__)
                status = getattr(exc, "status_code", None)
                if status == 429:
                    rate_limited = True
                    break
                if status in {400, 401, 403, 404}:
                    break
            finally:
                _audit(role, model, key, ok, start, "provider")
        if rate_limited:
            break
    if not api_key:
        _audit(role, primary, key_for(primary, system, user, temperature), False, started, "missing_key")
        raise LLMError("LLM_API_KEY is missing and the response is not cached.")
    raise LLMError(f"{role}: no schema-valid response ({type(last_error).__name__}).") from last_error
