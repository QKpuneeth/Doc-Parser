import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litellm import completion, completion_cost

load_dotenv()


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


VISION_ENABLED = _env("VISION_ENABLED", default="false").lower() == "true"
_AZURE_KEY = _env("AZURE_OPENAI_API_KEY")
_AZURE_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT").rstrip("/")
_AZURE_API_VERSION = _env("AZURE_OPENAI_API_VERSION")
_OPENAI_KEY = _env("OPENAI_API_KEY")
_OPENAI_ENDPOINT = _env("OPENAI_BASE_URL").rstrip("/")
_IS_AZURE = bool(_AZURE_KEY or _AZURE_ENDPOINT or _AZURE_API_VERSION)
VISION_MODEL = _env(
    "VISION_MODEL",
    default="AZURE_OPENAI_DEPLOYMENT_NAME" if _IS_AZURE else "gpt-4o-mini",
)
API_KEY = _AZURE_KEY or _OPENAI_KEY
ENDPOINT = _AZURE_ENDPOINT or _OPENAI_ENDPOINT
API_VERSION = _AZURE_API_VERSION
VISION_PROVIDER = "azure" if _IS_AZURE else "openai"

if API_KEY and ENDPOINT and VISION_MODEL:
    if _IS_AZURE:
        API_BASE = f"{ENDPOINT}/openai/deployments/{VISION_MODEL}"
    else:
        API_BASE = ENDPOINT
else:
    API_BASE = None

_ALLOWED_CATEGORIES = {"text_image", "diagram_chart", "photo", "other"}


def get_vision_client() -> dict[str, str] | None:
    if not VISION_ENABLED:
        return None
    if not API_KEY or not API_BASE:
        raise RuntimeError("Vision is enabled but its API key, endpoint, and model are not configured")
    if _IS_AZURE and not API_VERSION:
        raise RuntimeError("Azure Vision requires AZURE_OPENAI_API_VERSION")
    return {
        "api_key": API_KEY,
        "api_base": API_BASE,
        "api_version": API_VERSION,
        "model": VISION_MODEL,
        "provider": VISION_PROVIDER,
    }


def vision_available() -> bool:
    if not (VISION_ENABLED and API_KEY and API_BASE):
        return False
    return not _IS_AZURE or bool(API_VERSION)


def _result(path: Path, status: str, description: str = "", error_type: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": "other",
        "extracted_text": "",
        "description": description,
        "status": status,
        "image_path": path.as_posix(),
    }
    if error_type:
        result["error_type"] = error_type
    return result


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values = []
        for part in content:
            if isinstance(part, str):
                values.append(part)
            elif isinstance(part, dict):
                value = part.get("text")
                if value:
                    values.append(str(value))
        return "\n".join(values)
    return str(content or "")


def _parse_json_content(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except (TypeError, json.JSONDecodeError):
        return {
            "category": "other",
            "extracted_text": "",
            "description": cleaned,
        }
    if not isinstance(parsed, dict):
        return {
            "category": "other",
            "extracted_text": "",
            "description": cleaned,
        }
    return parsed


def _normalise_result(parsed: dict[str, Any], path: Path) -> dict[str, Any]:
    category = str(parsed.get("category", "other")).strip().lower()
    if category not in _ALLOWED_CATEGORIES:
        category = "other"
    result = {
        "category": category,
        "extracted_text": str(parsed.get("extracted_text") or "").strip(),
        "description": str(parsed.get("description") or "").strip(),
        "status": str(parsed.get("status") or "ok"),
        "image_path": path.as_posix(),
    }
    for key in ("usage_tokens", "cost_usd", "error_type"):
        if key in parsed:
            result[key] = parsed[key]
    return result


def classify_and_extract_image(
    image_path: str,
    enabled: bool = True,
    timeout: float = 60.0,
) -> dict[str, Any]:
    path = Path(image_path)
    if not enabled or not VISION_ENABLED:
        return _result(path, "disabled", "Vision disabled.")
    try:
        client = get_vision_client()
    except RuntimeError:
        return _result(path, "unavailable", error_type="RuntimeError")
    if client is None:
        return _result(path, "disabled", "Vision disabled.")

    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    except OSError as exc:
        return _result(path, "error", error_type=type(exc).__name__)

    prompt = """
Classify this document image into exactly one category: text_image, diagram_chart, photo, or other.
For text_image, transcribe all readable text faithfully.
For diagram_chart, describe the structure, labels, relationships, axes, legends, and important values, and transcribe important visible labels.
For photo, describe meaningful content and transcribe visible labels.
For other, briefly describe it.
Return only valid JSON with exactly these fields: category, extracted_text, description.
Do not infer values that are not visible.
""".strip()

    completion_kwargs: dict[str, Any] = {
        "model": (
            f"azure/{client['model']}"
            if client["provider"] == "azure"
            else client["model"]
        ),
        "api_key": client["api_key"],
        "api_base": client["api_base"],
        "temperature": 0,
        "timeout": float(timeout),
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encoded}",
                        },
                    },
                ],
            },
        ],
    }
    if client["provider"] == "azure":
        completion_kwargs["api_version"] = client["api_version"]
    try:
        response = completion(**completion_kwargs)
    except Exception as exc:
        return _result(path, "error", error_type=type(exc).__name__)

    content = _content_text(response.choices[0].message.content)
    parsed = _normalise_result(_parse_json_content(content), path)
    usage = getattr(response, "usage", None)
    parsed["usage_tokens"] = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = None
    parsed["cost_usd"] = round(float(cost), 6) if cost is not None else None
    return parsed
