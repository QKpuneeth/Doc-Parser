import argparse
import contextlib
import difflib
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
import warnings
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.backend_options import MsPowerpointBackendOptions, MsWordBackendOptions
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.types.doc import ContentLayer, DoclingDocument, PictureItem, TableItem

import image_signals
from vision_utils import classify_and_extract_image, vision_available


SUPPORTED = {".pdf", ".docx", ".pptx"}
ARCHIVE = {".zip"}
# Archive extraction is a trust boundary: an untrusted zip may try to escape the
# extraction directory (zip slip) or expand to an unbounded size (zip bomb). Only
# formats in SUPPORTED are ever written out, which bounds what lands on disk.
MAX_ARCHIVE_MEMBERS = 2000
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
ARCHIVE_CHUNK_BYTES = 1024 * 1024
IGNORED_ARCHIVE_DIRS = {"__MACOSX"}
IGNORED_ARCHIVE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
BACKGROUND_COVERAGE = 0.6
IMAGE_MARKER = "<!-- image -->"
PAGE_MARKER_RE = re.compile(r"<!-- PAGE: (\d+) -->")
IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
INLINE_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((data:[^)]*)\)")
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|media|var|tmp|opt|srv)/)")
SEMANTIC_LOSS_WARNING = "semantic_loss"
INFO_WARNING = "info"
WARNING_WARNING = "warning"
LOGGER = logging.getLogger("docling_slim")

PARSER_NAME = "docling-slim-parser"
SCHEMA_VERSION = "4.0"

# --- classification confidence bands (CHANGE 2) ------------------------------
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.45
# --- vision trigger threshold (CHANGE 3) -------------------------------------
DEFAULT_VISION_CONFIDENCE_THRESHOLD = 0.75
# --- full page duplicate detection (CHANGE 4) -------------------------------
FULL_PAGE_DUPLICATE_SIMILARITY = 0.9
FULL_PAGE_DUPLICATE_MIN_CHARACTERS = 120


@dataclass
class ParserConfig:
    """Single place for every tunable parser behaviour.

    Defaults preserve the behaviour that shipped before configuration was
    centralised, except for ``emit_canonical`` which follows the documented
    production layout (``document.md`` + ``metadata.json`` + ``manifest.json``
    + ``images/`` + ``logs/``) and can be re-enabled with ``--emit-canonical``.
    """

    # --- representation -----------------------------------------------------
    include_background: bool = False
    include_decorative: bool = False
    include_image_ocr: bool = True
    render_office_charts: bool = False
    images_scale: float = 2.0
    # --- vision (uncertainty driven) ---------------------------------------
    enable_vision: bool = False
    vision_max_images: int | None = None
    vision_timeout: float = 60.0
    vision_retry_count: int = 1
    vision_confidence_threshold: float = DEFAULT_VISION_CONFIDENCE_THRESHOLD
    # --- image policy -------------------------------------------------------
    enable_asset_deduplication: bool = True
    enable_full_page_duplicate_detection: bool = True
    full_page_duplicate_similarity: float = FULL_PAGE_DUPLICATE_SIMILARITY
    full_page_duplicate_min_characters: int = FULL_PAGE_DUPLICATE_MIN_CHARACTERS
    # --- output files -------------------------------------------------------
    emit_canonical: bool = False
    emit_metadata: bool = True
    validate_output: bool = True
    # --- runtime ------------------------------------------------------------
    force: bool = False
    log_level: str = "INFO"


def build_converter(
    do_ocr: bool = True,
    do_table_structure: bool = True,
    images_scale: float = 2.0,
    render_office_charts: bool = False,
) -> DocumentConverter:
    """Build a converter with the requested features enabled."""
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = do_ocr
    pdf_options.do_table_structure = do_table_structure
    pdf_options.generate_picture_images = True
    pdf_options.generate_table_images = True
    pdf_options.images_scale = float(images_scale)

    format_options: dict[InputFormat, Any] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
    }
    if render_office_charts:
        format_options[InputFormat.DOCX] = FormatOption(
            backend=MsWordDocumentBackend,
            pipeline_cls=SimplePipeline,
            backend_options=MsWordBackendOptions(render_chart_images=True),
        )
        format_options[InputFormat.PPTX] = FormatOption(
            backend=MsPowerpointDocumentBackend,
            pipeline_cls=SimplePipeline,
            backend_options=MsPowerpointBackendOptions(render_chart_images=True),
        )
    return DocumentConverter(format_options=format_options)


def _write_image_decision_log(records: list[dict[str, Any]], output_dir: Path) -> Path | None:
    """Write one machine-readable decision line per image.

    A human-readable log line answers "what happened"; this answers "why". Each
    line is a self-contained JSON object with the evidence, the confidence and
    the final decision, so a disputed image can be audited, and a threshold can
    be re-evaluated offline without re-running the parse.
    """
    if not records:
        return None
    path = output_dir / "logs" / "image_decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    {
                        "ref": record.get("ref"),
                        "page": record.get("page"),
                        "page_inferred": bool(record.get("page_inferred")),
                        "path": record.get("relative_path"),
                        "asset_sha256": record.get("sha256"),
                        "width": record.get("width"),
                        "height": record.get("height"),
                        "page_coverage": record.get("coverage"),
                        "ocr_characters": len(record.get("ocr") or ""),
                        "docling_class": record.get("classification"),
                        "semantic_class": record.get("semantic_class"),
                        "confidence": record.get("classification_confidence"),
                        "confidence_band": record.get("classification_confidence_band"),
                        "method": record.get("classification_method"),
                        "reason": record.get("classification_reason"),
                        "included_in_markdown": bool(record.get("include_in_md")),
                        "omit_reason": record.get("omit_reason"),
                        "duplicate_of": record.get("duplicate_of"),
                        "occurrences": len(record.get("duplicate_occurrences") or []) or 1,
                        "text_layer_duplicate": record.get("text_layer_duplicate"),
                        "vision_status": record.get("vision_status"),
                        "vision_skip_reason": record.get("vision_skip_reason"),
                        "vision_error": record.get("vision_error"),
                        "evidence": list(record.get("evidence") or []),
                        "decorative_evidence": list(record.get("decorative_evidence") or []),
                    },
                    ensure_ascii=False,
                    default=_json_safe,
                )
                + "\n"
            )
    LOGGER.info("Wrote per-image decision log for %s image(s)", len(records))
    return path


def _setup_logging(output_dir: Path, level: str) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        if getattr(handler, "_docling_slim_handler", False):
            LOGGER.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(log_dir / "parser.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    handler._docling_slim_handler = True
    LOGGER.addHandler(handler)


def _check_output(output_dir: Path, source_path: Path, force: bool) -> None:
    """Validate the destination without mutating it.

    The final directory is only created (atomically) once parsing, validation
    and manifest writing have all succeeded, so a failed run never leaves a
    half-populated output directory behind.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to replace it."
            )
        resolved_output = output_dir.resolve()
        resolved_source = source_path.resolve()
        if resolved_output == resolved_source or resolved_output in resolved_source.parents:
            raise ValueError("The output directory cannot contain the source document")


def _prepare_output(output_dir: Path, source_path: Path, force: bool) -> None:
    """Backward-compatible alias for :func:`_check_output`."""
    _check_output(output_dir, source_path, force)


def _make_working_dir(output_dir: Path) -> Path:
    """Create a sibling temporary directory on the same filesystem."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )


def _finalize_output(working_dir: Path, output_dir: Path) -> None:
    """Move a completed run into place, replacing any previous output."""
    for path in (working_dir / "images", working_dir / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    for handler in list(LOGGER.handlers):
        handler.flush()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(working_dir, output_dir)
    umask = os.umask(0o022)
    os.umask(umask)
    os.chmod(output_dir, 0o777 & ~umask)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _page_number(prov: Any) -> int | None:
    if prov is None:
        return None
    raw = getattr(prov, "page_no", None)
    if raw is None:
        raw = getattr(prov, "page", None)
    if raw is None:
        return None
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number if number >= 1 else number + 1


def _page_keys(doc: DoclingDocument) -> list[int]:
    pages = getattr(doc, "pages", {}) or {}
    if not hasattr(pages, "items"):
        return []
    result = []
    for key, _ in pages.items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        result.append(number if number >= 1 else number + 1)
    return sorted(set(result))


def _page_size(doc: DoclingDocument, page_no: int | None) -> tuple[float, float] | None:
    if page_no is None:
        return None
    pages = getattr(doc, "pages", {}) or {}
    if not hasattr(pages, "items"):
        return None
    page = None
    for key, value in pages.items():
        try:
            normalized = int(key)
        except (TypeError, ValueError):
            continue
        normalized = normalized if normalized >= 1 else normalized + 1
        if normalized == page_no:
            page = value
            break
    if page is None:
        return None
    size = getattr(page, "size", None)
    if size is None:
        return None
    try:
        if hasattr(size, "width") and hasattr(size, "height"):
            return float(size.width), float(size.height)
        return float(size[0]), float(size[1])
    except (TypeError, ValueError, IndexError):
        return None


def _bbox_tuple(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    try:
        values = bbox.as_tuple()
    except AttributeError:
        try:
            values = (bbox.l, bbox.t, bbox.r, bbox.b)
        except AttributeError:
            return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _provenance(item: Any) -> dict[str, Any]:
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return {"page": None, "bbox": None}
    first = provenance[0]
    return {"page": _page_number(first), "bbox": _bbox_tuple(getattr(first, "bbox", None))}


def _item_page(item: Any, page_count: int) -> int | None:
    page = _provenance(item)["page"]
    if page is not None:
        return page
    if page_count == 0:
        return 1
    return None


def _picture_coverage(doc: DoclingDocument, item: Any) -> float:
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return 0.0
    page_no = _page_number(provenance[0])
    size = _page_size(doc, page_no)
    bbox = _bbox_tuple(getattr(provenance[0], "bbox", None))
    if size is None or bbox is None:
        return 0.0
    left, top, right, bottom = bbox
    left, right = sorted((max(0.0, left), max(0.0, right)))
    top, bottom = sorted((max(0.0, top), max(0.0, bottom)))
    width = max(0.0, min(right, size[0]) - left)
    height = max(0.0, min(bottom, size[1]) - top)
    area = width * height
    page_area = size[0] * size[1]
    return area / page_area if page_area else 0.0


def _iter_items(doc: DoclingDocument) -> Iterable[tuple[Any, int]]:
    return doc.iterate_items(
        with_groups=False,
        traverse_pictures=True,
        included_content_layers=set(ContentLayer),
    )


def _item_lookup(doc: DoclingDocument) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    collection_names = (
        "texts",
        "pictures",
        "tables",
        "lists",
        "groups",
        "key_value_items",
        "form_items",
    )
    for name in collection_names:
        collection = getattr(doc, name, []) or []
        if not isinstance(collection, list):
            continue
        for item in collection:
            ref = getattr(item, "self_ref", None)
            if ref:
                lookup[str(ref)] = item
    return lookup


def _descendant_text(item: Any, lookup: Mapping[str, Any]) -> str:
    values: list[str] = []
    visited: set[str] = set()

    def visit(current: Any) -> None:
        for ref in getattr(current, "children", []) or []:
            ref_name = getattr(ref, "cref", None) or str(ref)
            if ref_name in visited:
                continue
            visited.add(ref_name)
            child = lookup.get(ref_name)
            if child is None:
                continue
            text = _as_text(getattr(child, "text", None))
            if text:
                values.append(text)
            visit(child)

    visit(item)
    return "\n".join(values)


def _refs_nested_in_pictures(
    doc: Any, lookup: Mapping[str, Any] | None = None
) -> set[str]:
    """self_refs of every item that lives inside a picture.

    The Markdown is exported with ``traverse_pictures=False``, so text nested in
    a picture is never emitted on its own: our own image block is what puts its
    words in the document. That makes the picture the only carrier of those
    words, and the page text layer therefore cannot be treated as a duplicate
    source for it.

    Picture children are ``RefItem`` references rather than nodes, so they are
    resolved through the document lookup exactly as ``_descendant_text`` does.
    """
    refs: set[str] = set()
    if lookup is None:
        lookup = _item_lookup(doc)
    pending: list[Any] = []
    for picture in getattr(doc, "pictures", None) or []:
        pending.extend(getattr(picture, "children", None) or [])
    while pending:
        entry = pending.pop()
        name = (
            getattr(entry, "cref", None)
            or getattr(entry, "self_ref", None)
            or str(entry)
        )
        if not name or name in refs:
            continue
        refs.add(name)
        node = entry if getattr(entry, "self_ref", None) else lookup.get(name)
        if node is not None:
            pending.extend(getattr(node, "children", None) or [])
    return refs


def _image_from_item(
    doc: DoclingDocument,
    item: Any,
    coverage: float,
    ocr_text: str,
) -> tuple[Image.Image | None, bool, str | None]:
    errors: list[str] = []
    try:
        image = item.get_image(doc)
        if image is not None:
            return image, False, None
    except Exception as exc:
        errors.append(type(exc).__name__)

    raw_image = getattr(item, "image", None)
    if raw_image is not None:
        image = getattr(raw_image, "pil_image", None)
        if image is not None:
            return image, False, None

    if coverage >= 0.5:
        page_no = _provenance(item)["page"]
        pages = getattr(doc, "pages", {}) or {}
        page = None
        if hasattr(pages, "items"):
            for key, value in pages.items():
                try:
                    normalized = int(key)
                except (TypeError, ValueError):
                    continue
                normalized = normalized if normalized >= 1 else normalized + 1
                if normalized == page_no:
                    page = value
                    break
        raw_page_image = getattr(page, "image", None) if page is not None else None
        page_image = getattr(raw_page_image, "pil_image", None)
        if page_image is not None:
            return page_image, True, None

    return None, False, ",".join(errors) if errors else "image unavailable"


def _image_hash(image: Image.Image) -> str:
    normalized = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(str(normalized.size).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _save_image_asset(
    image: Image.Image | None,
    image_dir: Path,
    prefix: str,
    existing_hashes: dict[str, str],
) -> tuple[str | None, str | None, str | None, int, int]:
    if image is None:
        return None, None, None, 0, 0
    try:
        image_hash = _image_hash(image)
        width, height = image.size
    except Exception as exc:
        return None, None, type(exc).__name__, 0, 0

    existing = existing_hashes.get(image_hash)
    if existing is not None:
        return existing, image_hash, None, width, height

    number = len(existing_hashes) + 1
    filename = f"{prefix}_{number:04d}.png"
    relative_path = f"images/{filename}"
    absolute_path = image_dir / filename
    try:
        image.convert("RGBA").save(absolute_path, format="PNG")
    except Exception as exc:
        return None, image_hash, type(exc).__name__, width, height
    existing_hashes[image_hash] = relative_path
    return relative_path, image_hash, None, width, height


def _link_duplicate_assets(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Group occurrences of the same asset and give each a canonical owner (CHANGE 5).

    Identical image content is stored once, so several Markdown entries can point
    at one file. The first occurrence owns the decision; later occurrences
    reference it through ``duplicate_of`` so a reader of the metadata can tell
    that a repeated logo on page 40 is the same asset as the one on page 1, not a
    new image. The shared decision is then propagated to every occurrence, which
    keeps a single asset from being classified differently just because it was
    seen on a page with different neighbours.
    """
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        digest = _as_text(record.get("sha256"))
        record.setdefault("asset_hash", digest)
        record.setdefault("duplicate_of", None)
        record.setdefault("duplicate_occurrences", [])
        if digest:
            by_hash.setdefault(digest, []).append(record)

    shared_fields = (
        "include_in_md",
        "omit_reason",
        "semantic_class",
        "classification",
        "classification_confidence",
        "classification_confidence_band",
        "classification_method",
        "classification_reason",
    )
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        owner = group[0]
        for record in group[1:]:
            record["duplicate_of"] = owner.get("relative_path") or owner.get("ref")
        for record in group:
            record["duplicate_occurrences"] = [
                {
                    "page": other.get("page"),
                    "ref": other.get("ref"),
                    "relative_path": other.get("relative_path"),
                    "is_owner": other is owner,
                }
                for other in group
            ]
        for field in shared_fields:
            if owner.get(field) is not None:
                for record in group[1:]:
                    record[field] = owner.get(field)

    return {
        "unique_assets": len(by_hash),
        "duplicate_assets": sum(1 for group in by_hash.values() if len(group) > 1),
        "duplicate_occurrences": sum(len(group) - 1 for group in by_hash.values() if len(group) > 1),
        "pages_per_asset": {
            digest: sorted({other.get("page") for other in group if other.get("page") is not None})
            for digest, group in sorted(by_hash.items())
        },
    }


_ALT_NOUNS = {
    image_signals.SEMANTIC_CHART: "chart",
    image_signals.SEMANTIC_DIAGRAM: "diagram",
    image_signals.SEMANTIC_TABLE: "table image",
    image_signals.SEMANTIC_LOGO: "logo",
    image_signals.SEMANTIC_PAGE_NUMBER: "page number",
    image_signals.SEMANTIC_DUPLICATE: "duplicate page image",
}

_SKIP_ALT_LINES = {"IQVIA", "PAGE", "P", "S", "1", "2", "3"}


def _clean_alt_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = value.replace("[", "(").replace("]", ")")
    value = value.strip("#*_` ")
    if len(value) > 140:
        value = value[:137].rstrip() + "..."
    return value


def _ocr_label(ocr: str) -> str:
    for line in ocr.splitlines():
        cleaned = _clean_alt_text(line)
        if len(cleaned) >= 4 and cleaned.upper() not in _SKIP_ALT_LINES:
            return cleaned
    return ""


def _image_alt(record: Mapping[str, Any], include_ocr: bool = True) -> str:
    """Build a descriptive (never invented) alt text for an image reference."""
    key = "alt_text" if include_ocr else "alt_text_plain"
    stored = _clean_alt_text(_as_text(record.get(key)))
    if stored:
        return stored
    ocr = _as_text(record.get("ocr")) if include_ocr else ""
    label = _ocr_label(ocr)
    noun = _ALT_NOUNS.get(_as_text(record.get("semantic_class")))
    if label:
        return f"{label} {noun}".strip() if noun else label
    vision = record.get("vision") or {}
    description = _clean_alt_text(_as_text(vision.get("description")))
    if description:
        return description
    classification = _as_text(record.get("semantic_class")).replace("_", " ")
    page = record.get("page")
    page_text = f"page {page}" if page is not None else "document"
    if classification:
        return f"{classification.capitalize()} on {page_text}"
    return f"Image on {page_text}"


def _blockquote(text: str, label: str) -> list[str]:
    values = text.splitlines() or [""]
    return [f"> **{label}:**"] + [f"> {line}" for line in values]


def _labelled_section(label: str, text: str) -> str:
    """Render ``**Label:**`` followed by plain paragraphs.

    Plain paragraphs (rather than block quotes) keep the text clean for
    downstream chunking and embedding, and the bold label keeps it unambiguous
    for human readers.
    """
    lines = [value.rstrip() for value in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    return "\n".join([f"**{label}:**", "", *lines])


# --- image transcription merge ------------------------------------------------
# Native OCR and the Vision model both transcribe the text physically present in
# an image, so rendering both as separate Markdown sections duplicates the same
# semantic content. The merge below chooses a single de-duplicated transcription
# for Markdown, while the rich inputs stay untouched for audit and future
# processing.

_TEXT_SOURCE_NONE = "none"
_TEXT_SOURCE_NATIVE_OCR = "native_ocr"
_TEXT_SOURCE_VISION = "vision"
_TEXT_SOURCE_MERGED = "merged"

# Vision transcription is treated as a duplicate of native OCR when similarity
# is at or above this threshold. Below it, only the genuinely new portion is
# appended to the OCR text.
_TRANSCRIPTION_DUPLICATE_SIMILARITY = 0.9


class _TranscriptionToken:
    """One whitespace token, compared case- and punctuation-insensitively."""

    __slots__ = ("norm", "original")

    def __init__(self, norm: str, original: str) -> None:
        self.norm = norm
        self.original = original

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TranscriptionToken) and self.norm == other.norm

    def __hash__(self) -> int:
        return hash(self.norm)


def _normalise_transcription(text: str) -> str:
    """Comparison-friendly form: Unicode-normalised, lowercased, punctuation-free."""
    value = unicodedata.normalize("NFKC", _as_text(text))
    value = re.sub(r"[\W_]+", " ", value)
    return " ".join(value.lower().split())


def _tokenise_transcription(text: str) -> list[_TranscriptionToken]:
    """Split transcription into tokens that ignore minor formatting differences."""
    tokens: list[_TranscriptionToken] = []
    for match in re.finditer(r"\S+", _as_text(text)):
        raw = match.group(0)
        norm = _normalise_transcription(raw)
        if norm:
            tokens.append(_TranscriptionToken(norm, raw))
    return tokens


def _merge_transcriptions(native_ocr: str, vision_text: str) -> dict[str, str]:
    """Choose the single transcription that should represent an image in Markdown.

    Returns ``{"text": ..., "source": ...}`` where ``source`` is one of
    ``none``, ``native_ocr``, ``vision`` or ``merged``. The rule is: never drop
    information, never duplicate it.

    * identical or near-identical content is reported once (``native_ocr``);
    * content that only Vision saw substitutes for empty native OCR (``vision``);
    * genuinely new Vision content is appended to the OCR text (``merged``).
    """
    ocr_tokens = _tokenise_transcription(native_ocr)
    vision_tokens = _tokenise_transcription(vision_text)
    if not ocr_tokens and not vision_tokens:
        return {"text": "", "source": _TEXT_SOURCE_NONE}
    if not ocr_tokens:
        return {"text": _as_text(vision_text).strip(), "source": _TEXT_SOURCE_VISION}
    if not vision_tokens:
        return {"text": _as_text(native_ocr).rstrip(), "source": _TEXT_SOURCE_NATIVE_OCR}

    matcher = difflib.SequenceMatcher(None, ocr_tokens, vision_tokens, autojunk=False)
    if matcher.ratio() >= _TRANSCRIPTION_DUPLICATE_SIMILARITY:
        return {"text": _as_text(native_ocr).rstrip(), "source": _TEXT_SOURCE_NATIVE_OCR}

    added: list[str] = []
    for _tag, _start_a, _end_a, start_b, end_b in matcher.get_opcodes():
        if _tag == "insert" or _tag == "replace":
            added.extend(token.original for token in vision_tokens[start_b:end_b])
    if not added:
        return {"text": _as_text(native_ocr).rstrip(), "source": _TEXT_SOURCE_NATIVE_OCR}
    return {
        "text": _as_text(native_ocr).rstrip() + "\n" + " ".join(added),
        "source": _TEXT_SOURCE_MERGED,
    }


def _semantic_text_for(record: Mapping[str, Any]) -> tuple[str, str]:
    """The deduplicated transcription for ``record`` plus its provenance.

    ``(text, source)`` is derived from the record's native OCR and Vision
    ``extracted_text``; the underlying fields are never mutated.
    """
    vision = record.get("vision") if isinstance(record.get("vision"), Mapping) else {}
    merged = _merge_transcriptions(
        _as_text(record.get("ocr")),
        _as_text(vision.get("extracted_text")),
    )
    return merged["text"], merged["source"]


def _image_block(record: Mapping[str, Any], include_ocr: bool = True) -> str:
    """Render one image as an asset reference plus labelled semantics.

    The Markdown keeps four clearly separated things:

    * ``![...](images/...)``      - the original image, always a relative path
    * ``**Image type:**``         - the deterministic classification
    * ``**OCR text:**``           - the transcription (native OCR, optionally
                                    extended or substituted by Vision text)
    * ``**Visual description:**`` - Vision output, only when Vision succeeded

    OCR and Vision transcription are merged into a single de-duplicated section
    so downstream chunking never sees the same content twice. Sections that have
    no content are omitted rather than filled with "not available" placeholders,
    and no parser/debug metadata is emitted.
    """
    path = record.get("relative_path") or record.get("path")
    if not path:
        return "<!-- image unavailable: the source image could not be extracted -->"
    sections = [f"![{_image_alt(record, include_ocr=include_ocr)}]({path})"]

    semantic_class = _as_text(record.get("semantic_class"))
    if semantic_class and semantic_class != image_signals.SEMANTIC_UNCLASSIFIED:
        sections.append(f"**Image type:** {image_signals.markdown_noun(semantic_class)}")

    semantic_text, _text_source = _semantic_text_for(record)
    if include_ocr and semantic_text:
        sections.append(_labelled_section("OCR text", semantic_text))

    vision = record.get("vision") or {}
    description = _as_text(vision.get("description"))
    if description:
        sections.append(_labelled_section("Visual description", description))
    return "\n\n".join(section for section in sections if section)


# Backwards-compatible name used by earlier versions of this module.
_vision_block = _image_block


def _render_markdown(
    markdown: str,
    records: list[dict[str, Any]],
    include_ocr: bool = True,
) -> str:
    parts = markdown.split(IMAGE_MARKER)
    marker_count = len(parts) - 1
    if marker_count != len(records):
        raise RuntimeError(
            f"Marker/picture count mismatch: {marker_count} markers vs {len(records)} pictures"
        )
    output = parts[0]
    for record, remainder in zip(records, parts[1:]):
        if record.get("include_in_md", True):
            block = _image_block(record, include_ocr=include_ocr)
        else:
            block = _omission_note(record)
        separator = "\n\n" if output.strip() else ""
        output = f"{output.rstrip()}{separator}{block}\n\n{remainder.lstrip(chr(10))}"
    while "\n\n\n" in output:
        output = output.replace("\n\n\n", "\n\n")
    return output.strip() + "\n"


def _omission_note(record: Mapping[str, Any]) -> str:
    """Traceable, chunker-invisible note for a deliberately omitted image.

    The reason is an HTML comment so neither humans reading the Markdown nor a
    downstream chunker mistake it for content; the machine-readable version of
    the same decision lives in metadata.json / canonical.json.
    """
    reason = record.get("omit_reason", "omitted")
    path = record.get("relative_path", "")
    return f"<!-- image omitted: {reason}; asset={path} -->"


def _page_markdown(doc: DoclingDocument, page_no: int) -> str:
    try:
        return doc.export_to_markdown(
            page_no=page_no,
            traverse_pictures=False,
            image_placeholder=IMAGE_MARKER,
            escape_html=False,
            escape_underscores=False,
        )
    except Exception as exc:
        LOGGER.warning("Page Markdown export failed for page %s: %s", page_no, type(exc).__name__)
        return ""


def _global_markdown(doc: DoclingDocument) -> str:
    return doc.export_to_markdown(
        traverse_pictures=False,
        image_placeholder=IMAGE_MARKER,
        escape_html=False,
        escape_underscores=False,
    )


def _escape_table_cell(value: Any) -> str:
    return _as_text(value).replace("|", "\\|").replace("\n", "<br>")


def _fallback_table_markdown(table: Mapping[str, Any]) -> str:
    rows = table.get("rows") or []
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    headers = list(table.get("headers") or [])
    if len(headers) < width:
        headers.extend(f"Column {index + 1}" for index in range(len(headers), width))
    headers = headers[:width]
    lines = [
        "| " + " | ".join(_escape_table_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = list(row) + [""] * (width - len(row))
        lines.append("| " + " | ".join(_escape_table_cell(value) for value in values[:width]) + " |")
    return "\n".join(lines)


def _table_markdown_block(table: Mapping[str, Any]) -> str:
    blocks: list[str] = []
    if not table.get("markdown"):
        fallback = _fallback_table_markdown(table)
        if fallback:
            blocks.append(fallback)
    if table.get("complex"):
        path = table.get("image_path")
        if path:
            blocks.append(f"![Original table image]({path})")
        else:
            blocks.append("> Original table image unavailable; structured rows are retained.")
    return "\n\n".join(blocks)


def _render_document_markdown(
    doc: DoclingDocument,
    records: list[dict[str, Any]],
    include_ocr: bool = True,
    tables: list[dict[str, Any]] | None = None,
) -> str:
    records = sorted(records, key=lambda record: int(record.get("order", 0)))
    tables = tables or []
    page_numbers = _page_keys(doc)
    if not page_numbers:
        global_markdown = _global_markdown(doc)
        rendered = _render_markdown(global_markdown, records, include_ocr=include_ocr)
        table_blocks = [
            _table_markdown_block(table)
            for table in tables
            if table.get("complex") or not table.get("markdown")
        ]
        if table_blocks:
            rendered = rendered.rstrip() + "\n\n" + "\n\n".join(table_blocks)
        return f"<!-- PAGE: 1 -->\n\n{rendered}"

    sections: list[str] = []
    known_pages = set(page_numbers)
    for page_no in page_numbers:
        page_records = [record for record in records if record.get("page") == page_no]
        page_tables = [table for table in tables if table.get("page") == page_no]
        page_md = _page_markdown(doc, page_no)
        rendered = _render_markdown(
            page_md,
            page_records,
            include_ocr=include_ocr,
        ) if page_md else ""
        table_blocks = [
            _table_markdown_block(table)
            for table in page_tables
            if table.get("complex") or not table.get("markdown")
        ]
        table_blocks = [block for block in table_blocks if block]
        if table_blocks and rendered:
            rendered = rendered.rstrip() + "\n\n" + "\n\n".join(table_blocks)
        elif table_blocks:
            rendered = "\n\n".join(table_blocks)
        section = f"<!-- PAGE: {page_no} -->"
        if rendered.strip():
            section += f"\n\n{rendered.strip()}"
        sections.append(section)

    unknown = [
        record
        for record in records
        if record.get("page") is not None and int(record.get("page")) not in known_pages
    ]
    no_page = [record for record in records if record.get("page") is None]
    extra_records = unknown + no_page
    if extra_records:
        blocks = []
        for record in extra_records:
            blocks.append(
                _image_block(record, include_ocr=include_ocr)
                if record.get("include_in_md", True)
                else f"<!-- IMAGE OMITTED: {record.get('omit_reason', 'omitted')}; asset={record.get('relative_path', '')} -->"
            )
        sections.append("<!-- PAGE: unknown -->\n\n" + "\n\n".join(blocks))
    return "\n\n".join(sections).strip() + "\n"


def _table_rows(item: TableItem, doc: DoclingDocument) -> tuple[list[str], list[list[str]]]:
    try:
        frame = item.export_to_dataframe(doc=doc)
        frame = frame.fillna("")
        headers = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.values.tolist()]
        return headers, rows
    except Exception as exc:
        LOGGER.warning("Table dataframe export failed: %s", type(exc).__name__)
        return [], []


def _table_image(
    item: TableItem,
    doc: DoclingDocument,
    image_dir: Path,
    hashes: dict[str, str],
    prefix: str,
) -> tuple[str | None, str | None, str | None]:
    try:
        image = item.get_image(doc)
    except Exception:
        image = None
    if image is None:
        raw_image = getattr(item, "image", None)
        image = getattr(raw_image, "pil_image", None)
    relative_path, image_hash, error, _, _ = _save_image_asset(image, image_dir, prefix, hashes)
    if error:
        LOGGER.warning("Table image extraction failed: %s", error)
    return relative_path, image_hash, error


def _picture_annotations(item: Any, meta: Any) -> tuple[list[str], list[str]]:
    """Collect annotation kinds and any text they carry, without deprecated fields."""
    kinds: list[str] = []
    texts: list[str] = []
    description = getattr(meta, "description", None)
    if description is not None:
        kinds.append("DescriptionAnnotation")
        text = _as_text(getattr(description, "text", ""))
        if text:
            texts.append(text)
    molecule = getattr(meta, "molecule", None)
    if molecule is not None and _as_text(getattr(molecule, "smi", "")):
        kinds.append("PictureMoleculeData")
    tabular_chart = getattr(meta, "tabular_chart", None)
    if tabular_chart is not None:
        kinds.append("PictureTabularChartData")
        title = _as_text(getattr(tabular_chart, "title", ""))
        if title:
            texts.append(title)
    extra = getattr(meta, "model_extra", None) or {}
    for key, value in extra.items():
        if not isinstance(value, Mapping):
            continue
        kind = str(key).rsplit(".", 1)[-1]
        if kind and kind not in kinds and ("chart" in kind or "annotation" in kind):
            kinds.append(kind)
            text = _as_text(value.get("title") or value.get("text") or value.get("description"))
            if text:
                texts.append(text)
    if not kinds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = getattr(item, "annotations", []) or []
        for annotation in legacy:
            kind = getattr(annotation, "kind", None)
            if kind:
                kinds.append(str(kind))
            text = _as_text(
                getattr(annotation, "text", "")
                or getattr(annotation, "description", "")
                or getattr(annotation, "title", "")
            )
            if text:
                texts.append(text)
    return kinds, texts


def _picture_metadata(item: Any) -> dict[str, Any]:
    meta = getattr(item, "meta", None)
    classification = None
    predictions = getattr(getattr(meta, "classification", None), "predictions", []) or []
    if predictions:
        classification = getattr(predictions[0], "class_name", None)
    annotations, descriptions = _picture_annotations(item, meta)
    tabular_chart = getattr(meta, "tabular_chart", None)
    chart_data = getattr(tabular_chart, "chart_data", None)
    if chart_data is None:
        return {
            "classification": str(classification) if classification else None,
            "annotations": annotations,
            "description": descriptions[0] if descriptions else None,
            "chart_data": None,
        }
    grid = getattr(chart_data, "grid", []) or []
    matrix = []
    for row in grid:
        values = []
        for cell in row:
            values.append(_as_text(getattr(cell, "text", "")))
        matrix.append(values)
    if not matrix:
        cells = getattr(chart_data, "table_cells", []) or []
        try:
            row_count = int(getattr(chart_data, "num_rows", 0) or 0)
            column_count = int(getattr(chart_data, "num_cols", 0) or 0)
        except (TypeError, ValueError):
            row_count = 0
            column_count = 0
        if row_count and column_count:
            matrix = [["" for _ in range(column_count)] for _ in range(row_count)]
            for cell in cells:
                try:
                    row = int(getattr(cell, "start_row_offset_idx"))
                    column = int(getattr(cell, "start_col_offset_idx"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if 0 <= row < row_count and 0 <= column < column_count:
                    matrix[row][column] = _as_text(getattr(cell, "text", ""))
    if not matrix:
        matrix = [[""]]
    headers = matrix[0]
    rows = matrix[1:] if len(matrix) > 1 else []
    return {
        "classification": str(classification) if classification else None,
        "annotations": annotations,
        "description": descriptions[0] if descriptions else None,
        "chart_data": {
            "title": _as_text(getattr(tabular_chart, "title", "")) or None,
            "headers": headers,
            "rows": rows,
        },
    }


def _classify_images(
    images: list[dict[str, Any]],
    config: ParserConfig | None = None,
    page_text: Mapping[Any, Sequence[str]] | None = None,
    page_size_by_page: Mapping[Any, tuple[float, float] | None] | None = None,
) -> dict[str, Any]:
    """Decide, per image, whether it carries information worth keeping in Markdown.

    Image geometry and OCR length alone are never decisive: the decision combines
    text, visual, positional and document-level signals, and defaults to keeping
    anything that is not positively identified as branding/decoration.
    """
    config = config or ParserConfig()
    context = image_signals.build_document_context(images)
    for record in images:
        full_page_duplicate = _full_page_duplicate_signals(
            record, page_text, config, page_size_by_page
        )
        if full_page_duplicate is not None:
            record["text_layer_duplicate"] = full_page_duplicate
        verdict = image_signals.classify_image(
            record, context, full_page_duplicate=full_page_duplicate
        )
        meaningful = bool(verdict["meaningful"])
        semantic_class = verdict["semantic_class"]
        record["semantic_class"] = semantic_class
        record["meaningful"] = meaningful
        record["classification_reason"] = verdict["classification_reason"]
        record["classification_confidence"] = verdict["classification_confidence"]
        record["classification_confidence_band"] = verdict["classification_confidence_band"]
        record["classification_method"] = verdict["classification_method"]
        record["evidence"] = list(verdict["meaningful_evidence"])
        record["meaningful_evidence"] = record["evidence"]
        record["decorative_evidence"] = verdict["decorative_evidence"]
        record["evidence_ledger"] = verdict["evidence_ledger"]
        record["signals"] = verdict["signals"]
        record["classification"] = record.get("classification") or semantic_class
        if not record.get("relative_path"):
            meaningful = False
        record["include_in_md"] = meaningful
        if meaningful:
            record.pop("omit_reason", None)
        else:
            record["include_in_md"] = bool(
                record.get("force_include_decorative") or record.get("include_decorative")
            )
            record["omit_reason"] = (
                record.get("omission_reason")
                or ("image_unavailable" if not record.get("relative_path") else "decorative_or_logo")
            )
        record.pop("force_include_decorative", None)
        record["alt_text"] = _image_alt(record)
        record["alt_text_plain"] = _image_alt(record, include_ocr=False)
    return context


FULL_PAGE_COVERAGE = 0.5


def _full_page_duplicate_signals(
    record: Mapping[str, Any],
    page_text: Mapping[Any, Sequence[str]] | None,
    config: ParserConfig,
    page_size_by_page: Mapping[Any, tuple[float, float] | None] | None = None,
) -> dict[str, Any] | None:
    """Compare a page-sized image against its page text layer (CHANGE 4).

    Only images that actually cover most of the page are compared, so ordinary
    figures are never at risk. Returns ``None`` when the check does not apply.
    """
    if not config.enable_full_page_duplicate_detection or not page_text:
        return None
    coverage = float(record.get("coverage") or 0.0)
    if coverage < FULL_PAGE_COVERAGE:
        return None
    if record.get("page_fallback"):
        # The asset is a rendering of the page itself, not an embedded picture.
        return None
    text = _as_text(record.get("ocr"))
    if not text.strip():
        return None
    page = record.get("page")
    page_layer = " ".join(page_text.get(page, ())) if page_text.get(page) else ""
    if not page_layer.strip():
        return None
    signals = image_signals.text_duplicate_signals(
        text,
        page_layer,
        similarity_threshold=config.full_page_duplicate_similarity,
        min_characters=config.full_page_duplicate_min_characters,
    )
    # Record the geometry behind the decision. When the asset has the same shape
    # as the page it is a page bitmap and its text is a pure transcript; when the
    # shapes differ it is a figure whose text was also exported as a text layer,
    # and the reader may reasonably want to keep seeing the figure.
    width = float(record.get("width") or 0.0)
    height = float(record.get("height") or 0.0)
    page_size = page_size_by_page.get(page) if page_size_by_page else None
    if width > 0 and height > 0 and page_size and page_size[0] and page_size[1]:
        image_ratio = width / height
        page_ratio = float(page_size[0]) / float(page_size[1])
        signals["page_aspect_match"] = bool(abs(image_ratio - page_ratio) <= 0.05 * page_ratio)
    if signals["duplicate"]:
        record["omission_reason"] = "full_page_image_duplicates_text_layer"
    return signals


def _is_decorative(record: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> bool:
    """Backwards-compatible wrapper around :func:`image_signals.classify_image`."""
    if record.get("include_background"):
        return False
    if context is None:
        context = image_signals.build_document_context([record])
    return not image_signals.classify_image(record, context)["meaningful"]


def _table_is_complex(item: TableItem) -> bool:
    data = getattr(item, "data", None)
    if data is None:
        return False
    try:
        if int(data.num_rows) > 30 or int(data.num_cols) > 6:
            return True
    except (AttributeError, TypeError, ValueError):
        pass
    for cell in getattr(data, "table_cells", []) or []:
        for field in ("start_row_offset", "end_row_offset", "start_col_offset", "end_col_offset"):
            try:
                if int(getattr(cell, field)) != 1:
                    return True
            except (AttributeError, TypeError, ValueError):
                continue
    return False


def _vision_priority(record: Mapping[str, Any]) -> float:
    """Rank images so that Vision calls are spent where they add information."""
    score = 0.0
    semantic_class = _as_text(record.get("semantic_class"))
    if semantic_class in {image_signals.SEMANTIC_CHART, image_signals.SEMANTIC_DIAGRAM}:
        score += 5.0
    elif semantic_class in {image_signals.SEMANTIC_TABLE, image_signals.SEMANTIC_TEXT}:
        score += 2.0
    text = (record.get("signals") or {}).get("text") or {}
    if text.get("has_percent") or text.get("distinct_numeric_count", 0) >= 2:
        score += 4.0
    if text.get("garbled"):
        score += 3.0
    if text.get("empty"):
        score += 2.0
    if float(record.get("coverage") or 0.0) >= 0.2:
        score += 1.0
    if record.get("chart_data"):
        score += 3.0
    if not record.get("include_in_md", True):
        score -= 100.0
    return round(score, 3)


TRANSIENT_VISION_ERRORS = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "ConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "HTTPStatusError",
        "InternalServerError",
        "PoolTimeout",
        "RateLimitError",
        "ReadTimeout",
        "RemoteProtocolError",
        "ServiceUnavailableError",
        "Timeout",
        "TimeoutError",
        "TooManyRequests",
    }
)
VISION_RETRY_BACKOFF_SECONDS = 1.0


def _vision_error_is_transient(error_type: Any) -> bool:
    """Whether a Vision failure is worth retrying (CHANGE 3).

    Only transport-level problems are retried. A malformed response or a bad
    request will fail identically on every attempt, so retrying it would only
    spend money and time.
    """
    name = _as_text(error_type)
    if not name:
        return False
    if name in TRANSIENT_VISION_ERRORS:
        return True
    lowered = name.lower()
    return any(marker in lowered for marker in ("timeout", "ratelimit", "rate_limit", "too many requests"))


def _apply_vision(
    records: list[dict[str, Any]],
    output_dir: Path,
    config: ParserConfig,
) -> dict[str, Any]:
    """Enrich meaningful images with Vision output; never fatal, always recorded."""
    calls = 0
    api_calls = 0
    errors = 0
    retried = 0
    skipped = 0
    cost = 0.0
    tokens = {"input": 0, "output": 0, "total": 0}
    statuses: Counter[str] = Counter()
    max_images = config.vision_max_images
    if max_images is not None and max_images < 0:
        raise ValueError("vision_max_images cannot be negative")
    threshold = float(config.vision_confidence_threshold)
    attempts_allowed = 1 + max(0, int(config.vision_retry_count))
    limit_skipped_priority: float | None = None

    for record in records:
        record["vision_priority"] = _vision_priority(record)

    if not config.enable_vision:
        for record in records:
            record["vision_status"] = "disabled"
            skipped += 1
            statuses["disabled"] += 1
        return {
            "calls": 0,
            "api_calls": 0,
            "errors": 0,
            "retried": 0,
            "skipped": skipped,
            "cost_usd": 0.0,
            "tokens": tokens,
            "statuses": dict(statuses),
            "confidence_threshold": threshold,
            "limit_reached": False,
            "limit_skipped_highest_priority": None,
        }
    if not vision_available():
        for record in records:
            record["vision_status"] = "unavailable"
            skipped += 1
            statuses["unavailable"] += 1
        return {
            "calls": 0,
            "api_calls": 0,
            "errors": 0,
            "retried": 0,
            "skipped": skipped,
            "cost_usd": 0.0,
            "tokens": tokens,
            "statuses": dict(statuses),
            "confidence_threshold": threshold,
            "limit_reached": False,
            "limit_skipped_highest_priority": None,
        }

    ordered = sorted(
        range(len(records)),
        key=lambda index: (-records[index]["vision_priority"], records[index].get("order", 0)),
    )
    for index in ordered:
        record = records[index]
        path = record.get("relative_path")
        if not path:
            record["vision_status"] = "unavailable"
            record["vision_error"] = "missing_asset"
            skipped += 1
            statuses["unavailable"] += 1
            continue
        if not record.get("include_in_md", True):
            record["vision_status"] = "skipped_decorative"
            skipped += 1
            statuses["skipped_decorative"] += 1
            continue
        # CHANGE 3: spend Vision calls only where the deterministic
        # classification is unsure. A confident classification already yields a
        # reliable image type, so the call would add cost without adding a
        # decision; images explicitly marked as required are always enriched.
        confidence = record.get("classification_confidence")
        if not record.get("vision_required") and isinstance(confidence, (int, float)):
            if float(confidence) >= threshold:
                record["vision_status"] = "skipped_confident"
                record["vision_skip_reason"] = (
                    f"classification confidence {float(confidence):.2f} is at or above "
                    f"the Vision threshold {threshold:.2f}"
                )
                skipped += 1
                statuses["skipped_confident"] += 1
                continue
        if max_images is not None and calls >= max_images:
            record["vision_status"] = "skipped_limit"
            record["vision_skip_reason"] = (
                f"Vision budget of {max_images} image(s) was already used by higher "
                "priority images"
            )
            priority = float(record.get("vision_priority") or 0.0)
            limit_skipped_priority = (
                priority if limit_skipped_priority is None else max(limit_skipped_priority, priority)
            )
            skipped += 1
            statuses["skipped_limit"] += 1
            continue
        calls += 1
        result: dict[str, Any] = {}
        for attempt in range(1, attempts_allowed + 1):
            api_calls += 1
            try:
                result = classify_and_extract_image(
                    str(output_dir / path),
                    enabled=True,
                    timeout=config.vision_timeout,
                )
            except Exception as exc:
                result = {"status": "error", "error_type": type(exc).__name__}
            error_type = _as_text(result.get("error_type")) if result else ""
            if not (result and result.get("status") == "error"):
                break
            if attempt < attempts_allowed and _vision_error_is_transient(error_type):
                retried += 1
                record["vision_retries"] = attempt
                LOGGER.warning(
                    "Vision attempt %d/%d for %s failed transiently (%s); retrying",
                    attempt,
                    attempts_allowed,
                    path,
                    error_type,
                )
                time.sleep(VISION_RETRY_BACKOFF_SECONDS * attempt)
                continue
            break
        result["image_path"] = path
        result.setdefault("attempts", attempt)
        record["vision"] = result
        record["vision_attempts"] = attempt
        status = _as_text(result.get("status")) or "ok"
        record["vision_status"] = status
        statuses[status] += 1
        if status == "error":
            errors += 1
            record["vision_error"] = _as_text(result.get("error_type")) or "vision_error"
            LOGGER.warning(
                "Vision failed for %s after %d attempt(s): %s",
                path,
                attempt,
                record["vision_error"],
            )
        cost += float(result.get("cost_usd") or 0.0)
        usage = result.get("usage_tokens") or {}
        for key, source in (("input", "input_tokens"), ("output", "output_tokens"), ("total", "total_tokens")):
            try:
                tokens[key] += int(usage.get(source) or 0)
            except (TypeError, ValueError):
                continue
        if status == "ok":
            record["alt_text"] = _image_alt(record)
            record["alt_text_plain"] = _image_alt(record, include_ocr=False)
    return {
        "calls": calls,
        "api_calls": api_calls,
        "errors": errors,
        "retried": retried,
        "skipped": skipped,
        "cost_usd": round(cost, 6),
        "tokens": tokens,
        "statuses": dict(statuses),
        "confidence_threshold": threshold,
        "limit_reached": limit_skipped_priority is not None,
        "limit_skipped_highest_priority": limit_skipped_priority,
    }


def _image_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, retrieval-oriented description of a single image occurrence.

    This is what a future RAG store needs to index an image without re-parsing
    the document or re-running classification: a stable id, where it came from,
    what it is and how sure we are, what text it carries, whether it made it
    into the Markdown, and whether it shares bytes with another occurrence.
    Deliberately not included: the raw signal payloads, evidence ledgers and
    Docling annotations that stay in ``canonical.json``.
    """
    path = record.get("relative_path") or record.get("path")
    raw_vision = record.get("vision")
    vision: Mapping[str, Any] = raw_vision if isinstance(raw_vision, Mapping) else {}
    occurrences = record.get("duplicate_occurrences") or []
    visual = record.get("visual") if isinstance(record.get("visual"), Mapping) else {}
    semantic_text, text_source = _semantic_text_for(record)
    return {
        "id": path,
        "path": path,
        "asset_sha256": record.get("sha256"),
        "page": record.get("page"),
        "pages": sorted({item.get("page") for item in occurrences if item.get("page") is not None})
        or ([record.get("page")] if record.get("page") is not None else []),
        "occurrences": len(occurrences) or 1,
        "duplicate_of": record.get("duplicate_of"),
        "type": record.get("semantic_class"),
        "docling_class": record.get("classification"),
        "confidence": record.get("classification_confidence"),
        "confidence_band": record.get("classification_confidence_band"),
        "classification_method": record.get("classification_method"),
        "classification_reason": record.get("classification_reason"),
        "included_in_markdown": bool(record.get("include_in_md")),
        "omission_reason": record.get("omit_reason"),
        "caption": record.get("alt_text"),
        "caption_plain": record.get("alt_text_plain"),
        "width": record.get("width") or visual.get("width"),
        "height": record.get("height") or visual.get("height"),
        "page_coverage": record.get("coverage"),
        "ocr_text": record.get("ocr") or "",
        "ocr_characters": len(record.get("ocr") or ""),
        "semantic_text": semantic_text,
        "text_source": text_source,
        "vision": (
            {
                "status": record.get("vision_status"),
                "category": vision.get("category"),
                "extracted_text": vision.get("extracted_text"),
                "description": vision.get("description"),
                "attempts": record.get("vision_attempts"),
                "error": record.get("vision_error"),
                "skipped_reason": record.get("vision_skip_reason"),
            }
            if (vision or record.get("vision_status"))
            else None
        ),
        "evidence": list(record.get("evidence") or record.get("meaningful_evidence") or []),
    }


def _table_metadata(table: Mapping[str, Any]) -> dict[str, Any]:
    rows = table.get("rows") or []
    headers = table.get("headers") or []
    width = len(headers) if headers else max((len(row) for row in rows), default=0)
    return {
        "id": table.get("ref"),
        "page": table.get("page"),
        "row_count": len(rows),
        "column_count": width,
        "complex": bool(table.get("complex")),
        "has_markdown": bool(table.get("markdown")),
        "has_image": bool(table.get("image_path")),
        "image_path": table.get("image_path"),
        "in_markdown": bool(table.get("in_markdown", True)),
    }


def _build_metadata(
    *,
    source_meta: Mapping[str, Any],
    source_path: Path,
    config: ParserConfig,
    started: datetime,
    pages: list[dict[str, Any]],
    images: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    asset_summary: Mapping[str, Any],
    vision_stats: Mapping[str, Any],
    warnings: list[dict[str, Any]],
    validation: Mapping[str, Any] | None,
    artifacts: Mapping[str, str | None],
    duplicate_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build ``metadata.json``: a document summary plus an image index (CHANGE 8/11)."""
    omitted = [record for record in images if not record.get("include_in_md")]
    by_class = Counter(record.get("semantic_class") or "unknown" for record in images)
    by_band = Counter(record.get("classification_confidence_band") or "unknown" for record in images)
    omission_reasons = Counter(
        record.get("omit_reason") or "unknown" for record in omitted if record.get("omit_reason")
    )
    text_characters = sum(
        int(page.get("text_characters") or 0) for page in pages
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "parser": {"name": PARSER_NAME, "version": SCHEMA_VERSION},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parse_started_at": started.isoformat(),
        "source": {
            "path": str(source_path),
            "filename": source_path.name,
            "format": source_path.suffix.lower().lstrip("."),
            "sha256": source_meta.get("sha256"),
            "bytes": source_meta.get("size_bytes"),
            "modified_at": source_meta.get("modified_at"),
        },
        "document": {
            "page_count": len(pages) or 1,
            "text_blocks": sum(int(page.get("text_blocks") or 0) for page in pages),
            "text_characters": text_characters,
            "tables": len(tables),
            "complex_tables": sum(1 for table in tables if table.get("complex")),
            "images": len(images),
            "unique_image_assets": asset_summary.get("unique_assets", len(images)),
            "duplicate_image_occurrences": asset_summary.get("duplicate_occurrences", 0),
        },
        "statistics": {
            "images_by_class": dict(sorted(by_class.items())),
            "images_by_confidence_band": dict(sorted(by_band.items())),
            "images_in_markdown": len(kept),
            "images_omitted": len(omitted),
            "omission_reasons": dict(sorted(omission_reasons.items())),
            "images_requiring_attention": [
                {
                    "image": record.get("relative_path") or record.get("path"),
                    "page": record.get("page"),
                    "type": record.get("semantic_class"),
                    "confidence": record.get("classification_confidence"),
                    "reason": record.get("classification_reason"),
                }
                for record in images
                if record.get("classification_confidence_band") == image_signals.BAND_LOW
            ],
            "vision": {
                "enabled": bool(config.enable_vision and vision_available()),
                "images_enriched": vision_stats.get("calls", 0),
                "api_calls": vision_stats.get("api_calls", vision_stats.get("calls", 0)),
                "errors": vision_stats.get("errors", 0),
                "retries": vision_stats.get("retried", 0),
                "skipped": vision_stats.get("skipped", 0),
                "statuses": vision_stats.get("statuses", {}),
                "confidence_threshold": vision_stats.get("confidence_threshold"),
                "limit_reached": vision_stats.get("limit_reached", False),
                "cost_usd": vision_stats.get("cost_usd", 0.0),
                "tokens": vision_stats.get("tokens", {}),
            },
        },
        "images": [_image_metadata(record) for record in images],
        # Counts only: the table content itself lives in document.md (and in
        # canonical.json when it is requested), so metadata.json stays a summary
        # rather than a second copy of the document.
        "tables": [_table_metadata(table) for table in tables],
        "pages": pages,
        "warnings": warnings,
        "duplicate_warnings": list(duplicate_warnings or []),
        "artifacts": dict(artifacts),
        "validation": dict(validation) if validation is not None else {"status": "pending"},
    }


def _raw_marker_counts(doc: DoclingDocument, records: list[dict[str, Any]]) -> dict[int | str, int]:
    pages = _page_keys(doc)
    if not pages:
        return {"global": _global_markdown(doc).count(IMAGE_MARKER)}
    result: dict[int | str, int] = {}
    for page in pages:
        result[page] = _page_markdown(doc, page).count(IMAGE_MARKER)
    unknown = sum(
        1
        for record in records
        if record.get("page") is None or int(record.get("page")) not in set(pages)
    )
    if unknown:
        result["unknown"] = unknown
    return result


def _validate_outputs(
    output_dir: Path,
    markdown: str,
    doc: DoclingDocument,
    records: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    config: ParserConfig | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    warning_details: list[dict[str, Any]] = []

    def warn(message: str, severity: str = INFO_WARNING) -> None:
        warnings.append(message)
        warning_details.append({"message": message, "severity": severity})

    expected_pages = _page_keys(doc) or [1]
    marker_pages = [int(value) for value in PAGE_MARKER_RE.findall(markdown)]
    if marker_pages != expected_pages:
        errors.append(
            f"Page markers mismatch: expected {expected_pages}, found {marker_pages}"
        )
    sequential = sorted(set(marker_pages))
    if marker_pages and marker_pages != sequential:
        errors.append(
            f"Page markers are duplicated or out of order: found {marker_pages}"
        )
    if IMAGE_MARKER in markdown:
        errors.append("Unreplaced Docling image marker remains in document.md")
    inline = INLINE_IMAGE_RE.findall(markdown)
    if inline:
        errors.append(
            f"Markdown embeds {len(inline)} inline base64 image(s); images must be external assets"
        )

    raw_counts = _raw_marker_counts(doc, records)
    if _page_keys(doc):
        expected_counts = Counter(
            record.get("page") if record.get("page") is not None else "unknown"
            for record in records
        )
    else:
        expected_counts = Counter({"global": len(records)})
    for page, count in raw_counts.items():
        if count != expected_counts.get(page, 0):
            errors.append(
                f"Image marker count mismatch on page {page}: {count} vs {expected_counts.get(page, 0)}"
            )

    references = Counter(
        reference
        for reference in IMAGE_REF_RE.findall(markdown)
        if reference.startswith("images/")
    )
    stray_references = [
        reference
        for reference in IMAGE_REF_RE.findall(markdown)
        if not reference.startswith("images/")
    ]
    for reference in stray_references:
        errors.append(f"Markdown image reference is not an external image asset: {reference}")
    expected_refs = Counter(
        record["relative_path"]
        for record in records
        if record.get("include_in_md", True) and record.get("relative_path")
    )
    for path, count in expected_refs.items():
        if references[path] < count:
            errors.append(f"Missing Markdown image reference for {path}")
        if not (output_dir / path).is_file():
            errors.append(f"Image asset is missing: {path}")
    for path in references:
        if not (output_dir / path).is_file():
            errors.append(f"Markdown points to missing image: {path}")

    for table in tables:
        representation = _table_representation(table)
        table["representation"] = representation
        if not table.get("markdown") and not table.get("rows"):
            warn(f"Table {table.get('ref', '?')} has no rendered representation")
        table_path = table.get("image_path")
        if table_path and not (output_dir / table_path).is_file():
            errors.append(f"Table image asset is missing: {table_path}")
        if table.get("complex") and not table_path:
            warn(
                f"Complex table {table.get('ref', '?')} has no extracted image",
                SEMANTIC_LOSS_WARNING,
            )
        if table.get("complex") and not table.get("rows") and not table_path:
            warn(
                f"Complex table {table.get('ref', '?')} has neither structured rows nor an image",
                SEMANTIC_LOSS_WARNING,
            )
    if not tables:
        warn("No tables were detected")
    if any(record.get("page") is None for record in records):
        warn("Some image records have no page provenance")
    if any(record.get("page_inferred") for record in records):
        warn("Some page assignments were inferred from document order")
    if any(record.get("image_error") for record in records):
        warn("Some source images could not be extracted")
    if any(
        record.get("include_in_md", True)
        and not _as_text(record.get("ocr"))
        and not _as_text((record.get("vision") or {}).get("description"))
        for record in records
    ):
        warn("Some images are included in Markdown without OCR or Vision text")

    semantic_loss = _semantic_loss_warnings(records, output_dir)
    for item in semantic_loss:
        warn(item["message"], SEMANTIC_LOSS_WARNING)

    # --- CHANGE 10: the artifacts a consumer will actually read --------------
    config = config or ParserConfig()
    markdown_path = output_dir / "document.md"
    if not markdown_path.is_file():
        errors.append("document.md was not written")
    elif markdown_path.stat().st_size == 0:
        errors.append("document.md is empty")

    # Paths must be relative so the output directory can be moved or zipped.
    for match in sorted(set(IMAGE_REF_RE.findall(markdown))):
        if match.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", match) or "://" in match:
            errors.append(f"Markdown image reference is not portable: {match}")

    for record in records:
        if not record.get("include_in_md", True):
            continue
        path = record.get("relative_path")
        if not path:
            errors.append(f"Image {record.get('ref', '?')} is kept but has no asset path")
            continue
        if path.startswith("/") or "://" in path:
            errors.append(f"Image asset path is not relative: {path}")
        if not (output_dir / path).is_file():
            errors.append(f"Kept image asset is missing from disk: {path}")

    # Every occurrence of one asset must carry one decision.
    by_asset: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        digest = _as_text(record.get("sha256"))
        if digest:
            by_asset.setdefault(digest, []).append(record)
    for digest, group in by_asset.items():
        decisions = {
            (bool(item.get("include_in_md")), item.get("semantic_class")) for item in group
        }
        if len(decisions) > 1:
            errors.append(
                f"Asset {group[0].get('relative_path')} is classified inconsistently across "
                f"{len(group)} occurrences: {sorted(str(item) for item in decisions)}"
            )

    # Confidence band must agree with the score that produced it.
    for record in records:
        confidence = record.get("classification_confidence")
        band = _as_text(record.get("classification_confidence_band"))
        if confidence is None or not band:
            continue
        if float(confidence) >= image_signals.CONFIDENCE_HIGH and band != image_signals.BAND_HIGH:
            errors.append(
                f"Image {record.get('relative_path')} has confidence {confidence} but band {band}"
            )
        elif (
            image_signals.CONFIDENCE_MEDIUM
            <= float(confidence)
            < image_signals.CONFIDENCE_HIGH
            and band != image_signals.BAND_MEDIUM
        ):
            errors.append(
                f"Image {record.get('relative_path')} has confidence {confidence} but band {band}"
            )
        elif float(confidence) < image_signals.CONFIDENCE_MEDIUM and band != image_signals.BAND_LOW:
            errors.append(
                f"Image {record.get('relative_path')} has confidence {confidence} but band {band}"
            )

    metadata_path = output_dir / "metadata.json"
    if config.emit_metadata:
        if not metadata_path.is_file():
            errors.append("metadata.json was not written although metadata output is enabled")
        else:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"metadata.json is not valid JSON: {type(exc).__name__}")
            else:
                for image in metadata.get("images") or []:
                    image_path = image.get("path")
                    if not image_path:
                        continue
                    if image_path.startswith("/") or "://" in image_path:
                        errors.append(f"metadata.json image path is not relative: {image_path}")
                    if not (output_dir / image_path).is_file():
                        errors.append(f"metadata.json references a missing image: {image_path}")
                    if bool(image.get("included_in_markdown")) != references.get(image_path, 0) > 0:
                        errors.append(
                            f"metadata.json disagrees with document.md about {image_path}"
                        )
    elif metadata_path.is_file():
        errors.append("metadata.json exists although metadata output is disabled")

    canonical_path = output_dir / "canonical.json"
    if config.emit_canonical:
        if not canonical_path.is_file():
            errors.append("canonical.json was not written although canonical output is enabled")
        else:
            try:
                json.loads(canonical_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"canonical.json is not valid JSON: {type(exc).__name__}")
    elif canonical_path.is_file():
        errors.append("canonical.json exists although canonical output is disabled")

    has_error = bool(errors)
    has_semantic_loss = any(
        detail["severity"] == SEMANTIC_LOSS_WARNING for detail in warning_details
    )
    status = "fail" if has_error else ("warning" if has_semantic_loss else "pass")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "warning_details": warning_details,
        "semantic_loss": [item for item in semantic_loss],
        "page_markers": marker_pages,
        "raw_image_markers": raw_counts,
        "image_references": dict(references),
    }


def _table_representation(table: Mapping[str, Any]) -> str:
    has_markdown = bool(table.get("markdown"))
    has_rows = bool(table.get("rows"))
    has_image = bool(table.get("image_path"))
    if has_markdown and has_image:
        return "structured+image"
    if has_markdown or has_rows:
        return "structured+image" if has_image else "structured"
    return "image" if has_image else "none"


def _structured_chart_data(record: Mapping[str, Any]) -> bool:
    chart_data = record.get("chart_data")
    if not isinstance(chart_data, Mapping):
        return False
    return bool(chart_data.get("rows") or chart_data.get("headers"))


def _type_aware_warnings(
    records: list[dict[str, Any]],
    config: ParserConfig,
) -> list[dict[str, Any]]:
    """Bucket image findings by type and severity (CHANGE 6).

    A "chart we could not read" and "a logo was left out" are not the same kind of
    problem, and neither is a document where five percent of the images are
    uncertain. Reporting them in one undifferentiated list forces a reader to
    re-derive the severity for every line, so each finding is labelled with the
    image type, the severity it deserves, and what would resolve it.
    """
    findings: list[dict[str, Any]] = []
    for record in records:
        semantic_class = _as_text(record.get("semantic_class")) or "unknown"
        path = record.get("relative_path") or record.get("path")
        page = record.get("page")
        confidence = record.get("classification_confidence")
        band = _as_text(record.get("classification_confidence_band"))
        included = bool(record.get("include_in_md"))

        if included and band == image_signals.BAND_LOW:
            findings.append(
                {
                    "severity": WARNING_WARNING,
                    "kind": "uncertain_classification",
                    "type": semantic_class,
                    "image": path,
                    "page": page,
                    "message": (
                        f"{path} on page {page} was kept as {semantic_class} but the "
                        f"classification is only {confidence} confident; enable Vision or "
                        "review the image before relying on the type."
                    ),
                    "suggestion": "re-run with --vision to have the type confirmed by a model",
                }
            )
        elif included and band == image_signals.BAND_MEDIUM:
            findings.append(
                {
                    "severity": INFO_WARNING,
                    "kind": "medium_confidence_classification",
                    "type": semantic_class,
                    "image": path,
                    "page": page,
                    "message": (
                        f"{path} on page {page} was classified as {semantic_class} with "
                        f"{confidence} confidence."
                    ),
                    "suggestion": None,
                }
            )

        if not included and semantic_class in image_signals.CONTENT_IMAGE_CLASSES:
            if _structured_chart_data(record):
                continue
            indicators = _semantic_indicators(record)
            if not indicators:
                continue
            findings.append(
                {
                    "severity": WARNING_WARNING,
                    "kind": "content_image_omitted",
                    "type": semantic_class,
                    "image": path,
                    "page": page,
                    "message": (
                        f"{path} on page {page} looks like {semantic_class} content but was "
                        f"omitted from the Markdown ({', '.join(indicators)})."
                    ),
                    "suggestion": (
                        "re-run with --include-decorative or raise the classification "
                        "thresholds if the omission is wrong"
                    ),
                }
            )
        elif not included and semantic_class == image_signals.SEMANTIC_LOGO:
            findings.append(
                {
                    "severity": INFO_WARNING,
                    "kind": "branding_omitted",
                    "type": semantic_class,
                    "image": path,
                    "page": page,
                    "message": f"Branding asset {path} on page {page} was omitted.",
                    "suggestion": None,
                }
            )

    if records:
        low = sum(
            1
            for record in records
            if _as_text(record.get("classification_confidence_band")) == image_signals.BAND_LOW
        )
        share = low / len(records)
        if share > 0.2:
            findings.append(
                {
                    "severity": WARNING_WARNING,
                    "kind": "low_confidence_dominates",
                    "type": None,
                    "image": None,
                    "page": None,
                    "message": (
                        f"{low} of {len(records)} images ({share:.0%}) have a low confidence "
                        "classification; the document is probably scanned or unusually visual."
                    ),
                    "suggestion": "enable Vision, or check that the PDF has an extractable text layer",
                }
            )
        if config.enable_vision and config.vision_max_images is not None:
            enriched = sum(1 for record in records if _as_text(record.get("vision_status")) == "ok")
            uncertain = low + sum(
                1
                for record in records
                if _as_text(record.get("classification_confidence_band"))
                == image_signals.BAND_MEDIUM
            )
            if uncertain > enriched:
                findings.append(
                    {
                        "severity": INFO_WARNING,
                        "kind": "vision_budget_exhausted",
                        "type": None,
                        "image": None,
                        "page": None,
                        "message": (
                            f"Vision enriched {enriched} image(s) while {uncertain} were "
                            f"uncertain; --vision-max-images {config.vision_max_images} was reached."
                        ),
                        "suggestion": "raise --vision-max-images to confirm the remaining images",
                    }
                )
    return findings


def _semantic_indicators(record: Mapping[str, Any]) -> list[str]:
    """Independent scan for information-bearing content in an image record.

    This deliberately re-derives the indicators from OCR text and metadata rather
    than trusting the classifier verdict, so that a policy mistake (or a future
    stricter policy) can still surface as a semantic-loss warning. Content that is
    preserved structurally (chart data, table rows) is not treated as a loss.
    """
    text = _as_text(record.get("ocr"))
    signals = image_signals.text_signals(text, [])
    indicators: list[str] = []
    if signals["has_percent"]:
        indicators.append("quantitative OCR content (percentages)")
    if signals["has_currency"] or signals["has_units"]:
        indicators.append("measured values with units or currency")
    if signals["distinct_numeric_count"] >= 2:
        indicators.append("multiple numeric values in OCR")
    if signals["content_word_count"] >= 2:
        indicators.append("meaningful OCR text")
    if signals["chart_terms"] or signals["chart_phrases"] or signals["diagram_terms"]:
        indicators.append("chart or diagram vocabulary in OCR")
    if signals["table_like"]:
        indicators.append("table-like structure in OCR")
    semantic_class = _as_text(record.get("semantic_class"))
    if semantic_class in {
        image_signals.SEMANTIC_CHART,
        image_signals.SEMANTIC_DIAGRAM,
        image_signals.SEMANTIC_TABLE,
    }:
        indicators.append(f"classified as {semantic_class.replace('_', ' ')}")
    classification = _as_text(record.get("classification")).lower()
    if any(term in classification for term in ("chart", "graph", "diagram", "plot")):
        indicators.append("Docling chart/diagram classification")
    if record.get("chart_data") and not _structured_chart_data(record):
        indicators.append("chart metadata without extracted values")
    vision = record.get("vision") or {}
    if _as_text(vision.get("description")) or _as_text(vision.get("extracted_text")):
        indicators.append("Vision description available")
    return sorted(set(indicators))


def _semantic_loss_warnings(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Report images omitted from Markdown that plausibly carried information."""
    findings: list[dict[str, Any]] = []
    for record in records:
        if record.get("include_in_md", True):
            continue
        if _structured_chart_data(record):
            continue
        if record.get("omit_reason") == "full_page_image_duplicates_text_layer":
            # Not a loss of information: the page text layer already carries this
            # content, so the Markdown does not drop anything the reader could not
            # read. The omission is still counted and reported separately.
            continue
        indicators = _semantic_indicators(record)
        if not indicators:
            continue
        path = record.get("relative_path") or record.get("path")
        name = Path(str(path)).name if path else f"image {record.get('ref', '?')}"
        page = record.get("page")
        page_text = f"on page {page}" if page is not None else "on an unknown page"
        findings.append(
            {
                "image": name,
                "relative_path": path,
                "page": page,
                "reason": record.get("omit_reason"),
                "evidence": indicators,
                "message": (
                    f"Potentially meaningful image {name} {page_text} was omitted from Markdown "
                    f"because it contains {', '.join(indicators)}."
                ),
            }
        )
    return findings


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_safe) + "\n",
        encoding="utf-8",
    )


def _page_stats(
    doc: DoclingDocument,
    text_blocks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pages = _page_keys(doc) or [1]
    result = []
    for page_no in pages:
        page_text = [block for block in text_blocks if block.get("page") == page_no]
        page_tables = [table for table in tables if table.get("page") == page_no]
        page_images = [image for image in images if image.get("page") == page_no]
        size = _page_size(doc, page_no)
        result.append(
            {
                "page": page_no,
                "width": size[0] if size else None,
                "height": size[1] if size else None,
                "text_blocks": len(page_text),
                "text_characters": sum(len(block.get("text", "")) for block in page_text),
                "tables": len(page_tables),
                "images": len(page_images),
            }
        )
    return result


def parse_document(
    input_path: str,
    output_dir: str = "output_docling",
    include_background: bool = False,
    include_decorative: bool = False,
    include_image_ocr: bool = True,
    render_office_charts: bool = False,
    enable_vision: bool | None = False,
    vision_max_images: int | None = None,
    vision_timeout: float = 60.0,
    images_scale: float = 2.0,
    force: bool = False,
    log_level: str = "INFO",
    # --- added options; all default to the documented production behaviour ---
    vision_retry_count: int = 1,
    vision_confidence_threshold: float = DEFAULT_VISION_CONFIDENCE_THRESHOLD,
    enable_asset_deduplication: bool = True,
    enable_full_page_duplicate_detection: bool = True,
    full_page_duplicate_similarity: float = FULL_PAGE_DUPLICATE_SIMILARITY,
    full_page_duplicate_min_characters: int = FULL_PAGE_DUPLICATE_MIN_CHARACTERS,
    emit_canonical: bool = False,
    emit_metadata: bool = True,
    validate_output: bool = True,
    config: ParserConfig | None = None,
) -> dict[str, Any]:
    source_path = Path(input_path)
    if source_path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported format: {source_path.suffix}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if vision_max_images is not None and vision_max_images < 0:
        raise ValueError("vision_max_images cannot be negative")
    if images_scale <= 0:
        raise ValueError("images_scale must be positive")
    if vision_retry_count < 0:
        raise ValueError("vision_retry_count cannot be negative")
    if not 0.0 <= float(vision_confidence_threshold) <= 1.0:
        raise ValueError("vision_confidence_threshold must be between 0 and 1")
    if not 0.0 <= float(full_page_duplicate_similarity) <= 1.0:
        raise ValueError("full_page_duplicate_similarity must be between 0 and 1")

    if config is None:
        config = ParserConfig(
            include_background=include_background,
            include_decorative=include_decorative,
            include_image_ocr=include_image_ocr,
            render_office_charts=render_office_charts,
            enable_vision=False if enable_vision is None else enable_vision,
            vision_max_images=vision_max_images,
            vision_timeout=float(vision_timeout),
            vision_retry_count=int(vision_retry_count),
            vision_confidence_threshold=float(vision_confidence_threshold),
            images_scale=float(images_scale),
            enable_asset_deduplication=bool(enable_asset_deduplication),
            enable_full_page_duplicate_detection=bool(enable_full_page_duplicate_detection),
            full_page_duplicate_similarity=float(full_page_duplicate_similarity),
            full_page_duplicate_min_characters=int(full_page_duplicate_min_characters),
            emit_canonical=bool(emit_canonical),
            emit_metadata=bool(emit_metadata),
            validate_output=bool(validate_output),
            force=force,
            log_level=log_level,
        )
    output_path = Path(output_dir)
    _check_output(output_path, source_path, config.force)
    working_path = _make_working_dir(output_path)
    (working_path / "images").mkdir(parents=True, exist_ok=True)
    (working_path / "logs").mkdir(parents=True, exist_ok=True)
    _setup_logging(working_path, config.log_level)
    started = datetime.now(timezone.utc)
    source_meta = _source_metadata(source_path, started)
    LOGGER.info("Starting conversion for %s", source_path)
    LOGGER.info("Source sha256=%s size=%s", source_meta["sha256"], source_meta["size_bytes"])
    try:
        canonical = _parse_into(
            source_path=source_path,
            working_path=working_path,
            config=config,
            source_meta=source_meta,
            started=started,
        )
    except BaseException:
        _discard_working_dir(working_path)
        raise
    _finalize_output(working_path, output_path)
    LOGGER.info("Output written to %s", output_path)
    for handler in list(LOGGER.handlers):
        handler.flush()
    _refresh_manifest_checksums(output_path)
    return canonical


# --- batch and archive inputs -------------------------------------------------
#
# ``parse_document`` is deliberately left untouched: it parses exactly one file
# into exactly one directory, and the rest of the module (including its tests)
# depends on that contract. Everything below only decides *which* files to feed
# it and *where* each result should land, so single-document behaviour cannot
# drift.


def _is_within(child: Path, parent: Path) -> bool:
    """Whether ``child`` really resolves inside ``parent``."""
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _archive_member_is_noise(name: str) -> bool:
    """Skip OS metadata that archives of documents habitually carry."""
    parts = PurePosixPath(name).parts
    if not parts:
        return True
    if IGNORED_ARCHIVE_DIRS.intersection(parts):
        return True
    return parts[-1] in IGNORED_ARCHIVE_NAMES


def _copy_bounded(source: Any, sink: Any, limit: int) -> int:
    """Copy at most ``limit`` bytes, raising if the stream exceeds it.

    The size recorded in a zip header is attacker-controlled, so the cap is
    enforced against the bytes actually read rather than the declared size.
    """
    copied = 0
    while True:
        chunk = source.read(ARCHIVE_CHUNK_BYTES)
        if not chunk:
            return copied
        copied += len(chunk)
        if copied > limit:
            raise ValueError(f"Archive expands past the {limit} byte limit")
        sink.write(chunk)


def _extract_archive(archive_path: Path, destination: Path) -> list[Path]:
    """Extract the supported documents from a zip into ``destination``.

    Only members whose suffix is in ``SUPPORTED`` are written, so a hostile
    archive cannot drop executables or arbitrary files into the extraction
    directory. Absolute paths and ``..`` traversal are refused outright, and both
    the declared and the actual uncompressed size are capped.
    """
    extracted: list[Path] = []
    budget = MAX_ARCHIVE_BYTES
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(
                f"Archive has {len(members)} members, above the {MAX_ARCHIVE_MEMBERS} limit: "
                f"{archive_path}"
            )
        for member in members:
            if member.is_dir() or _archive_member_is_noise(member.filename):
                continue
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                LOGGER.warning("Refusing unsafe archive member: %s", member.filename)
                continue
            if Path(member.filename).suffix.lower() not in SUPPORTED:
                continue
            if member.file_size > budget:
                raise ValueError(
                    f"Archive expands past the {MAX_ARCHIVE_BYTES} byte limit: {archive_path}"
                )
            target = destination / relative
            if not _is_within(target.parent, destination):
                LOGGER.warning("Refusing archive member outside the extraction root: %s", member.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(member) as source, target.open("wb") as sink:
                    budget -= _copy_bounded(source, sink, budget)
            except ValueError:
                target.unlink(missing_ok=True)
                raise
            extracted.append(target)
    return sorted(extracted, key=lambda path: path.as_posix())


def _documents_under(directory: Path) -> list[Path]:
    """Every supported document inside ``directory``, recursively."""
    found: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(directory)
        if any(part.startswith(".") or part in IGNORED_ARCHIVE_DIRS for part in relative.parts[:-1]):
            continue
        found.append(path)
    return found


@contextlib.contextmanager
def _collect_documents(inputs: Sequence[str | Path]) -> Iterator[list[Path]]:
    """Expand files, directories and zips into a flat, de-duplicated document list.

    Temporary extractions are removed when the context exits, so the caller never
    has to manage archive lifetimes.
    """
    with contextlib.ExitStack() as stack:
        documents: list[Path] = []
        for raw in inputs:
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(path)
            if path.is_dir():
                documents.extend(_documents_under(path))
                continue
            suffix = path.suffix.lower()
            if suffix in ARCHIVE:
                temporary = Path(
                    stack.enter_context(tempfile.TemporaryDirectory(prefix="docling-slim-archive-"))
                )
                documents.extend(_extract_archive(path, temporary))
                continue
            if suffix not in SUPPORTED:
                raise ValueError(f"Unsupported format: {path.suffix}")
            documents.append(path)
        unique: dict[str, Path] = {}
        for document in documents:
            unique.setdefault(str(document.resolve()), document)
        ordered = sorted(unique.values(), key=lambda item: item.as_posix())
        if not ordered:
            raise ValueError(
                "No documents found. Expected a PDF/DOCX/PPTX file, a directory containing "
                "them, or a .zip archive of them."
            )
        yield ordered


def _unique_output_name(source: Path, taken: set[str]) -> str:
    """A readable, collision-free directory name for one document's output."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "document"
    name = base
    counter = 2
    while name in taken:
        name = f"{base}_{counter}"
        counter += 1
    taken.add(name)
    return name


def _batch_totals(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    cost = 0.0
    for entry in results:
        stats = entry.get("stats") or {}
        for key in ("pages", "text_blocks", "tables", "images_total", "images_in_md"):
            try:
                totals[key] += int(stats.get(key) or 0)
            except (TypeError, ValueError):
                continue
        for key in ("vision_api_calls", "vision_errors", "vision_retries"):
            try:
                totals[key] += int(stats.get(key) or 0)
            except (TypeError, ValueError):
                continue
        try:
            cost += float(stats.get("vision_cost_usd") or 0.0)
        except (TypeError, ValueError):
            continue
    totals["vision_cost_usd"] = round(cost, 6)
    return dict(totals)


# Derived from the live signature so a new parse_document option is accepted by
# parse_many automatically, and a typo is still rejected loudly.
_PARSE_DOCUMENT_OPTIONS = frozenset(
    inspect.signature(parse_document).parameters
) - {"input_path", "output_dir", "config"}


def parse_many(
    inputs: str | Path | Sequence[str | Path],
    output_dir: str = "output_docling",
    continue_on_error: bool = True,
    **options: Any,
) -> dict[str, Any]:
    """Parse every document in ``inputs`` and report one row per document.

    ``inputs`` accepts a mix of document paths, directories (walked recursively)
    and ``.zip`` archives. Layout follows the number of documents discovered: a
    single document writes straight into ``output_dir`` exactly as
    :func:`parse_document` does, and two or more each get their own subdirectory
    named after the source file. Colliding names are suffixed rather than merged.

    By default a document that fails is recorded and the rest continue, because
    one unreadable study should not discard a whole batch. Set
    ``continue_on_error=False`` to stop at the first failure instead.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    requested = list(inputs)
    if not requested:
        raise ValueError("At least one input is required")
    unknown = sorted(set(options) - _PARSE_DOCUMENT_OPTIONS)
    if unknown:
        raise TypeError(f"Unknown option(s) for parse_many: {', '.join(unknown)}")

    output_root = Path(output_dir)
    with _collect_documents(requested) as documents:
        taken: set[str] = set()
        targets = [
            (document, output_root / _unique_output_name(document, taken))
            for document in documents
        ]
        if len(targets) == 1:
            # One document keeps the historical layout: its files land directly in
            # output_dir, so `parse_many([single])` is indistinguishable from
            # `parse_document(single)`.
            targets = [(targets[0][0], output_root)]
        LOGGER.info("Batch resolved %d document(s) from %d input(s)", len(targets), len(requested))
        results: list[dict[str, Any]] = []
        for position, (source, destination) in enumerate(targets, start=1):
            entry: dict[str, Any] = {
                "index": position,
                "source": source.name,
                "source_path": str(source),
                "output": str(destination),
                "status": "ok",
                "validation": None,
                "error": None,
            }
            LOGGER.info("Batch [%d/%d] Parsing %s", position, len(targets), source)
            try:
                canonical = parse_document(str(source), str(destination), **options)
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                LOGGER.error("Batch [%d/%d] %s failed: %s", position, len(targets), source, entry["error"])
                results.append(entry)
                if not continue_on_error:
                    raise
                continue
            entry["validation"] = (canonical.get("validation") or {}).get("status")
            entry["stats"] = canonical.get("stats") or {}
            results.append(entry)

    failed = [entry for entry in results if entry["status"] != "ok"]
    return {
        "documents": len(results),
        "succeeded": len(results) - len(failed),
        "failed": len(failed),
        "output": str(output_root),
        "totals": _batch_totals(results),
        "results": results,
    }


def _refresh_manifest_checksums(output_path: Path) -> None:
    """Recompute artifact checksums once logging has stopped.

    The log file keeps growing after the manifest is first written, so its
    size and digest are refreshed here to keep the manifest verifiable.
    """
    manifest_path = output_path / "manifest.json"
    if not manifest_path.is_file():
        return
    for handler in list(LOGGER.handlers):
        handler.flush()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest.get("artifacts", []):
        target = output_path / artifact["path"]
        if target.is_file():
            artifact["bytes"] = target.stat().st_size
            artifact["sha256"] = _sha256_file(target)
    _write_json(manifest_path, manifest)


def _discard_working_dir(working_path: Path) -> None:
    for handler in list(LOGGER.handlers):
        handler.flush()
    shutil.rmtree(working_path, ignore_errors=True)


def _source_metadata(source_path: Path, started: datetime) -> dict[str, Any]:
    return {
        "filename": source_path.name,
        "path": str(source_path),
        "format": source_path.suffix.lower().lstrip("."),
        "size_bytes": source_path.stat().st_size,
        "sha256": _sha256_file(source_path),
        "modified_at": datetime.fromtimestamp(
            source_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "parsed_at": started.isoformat(),
    }


def _parse_into(
    source_path: Path,
    working_path: Path,
    config: ParserConfig,
    source_meta: Mapping[str, Any],
    started: datetime,
) -> dict[str, Any]:
    output_path = working_path
    if config.render_office_charts and not shutil.which("soffice"):
        LOGGER.warning("Office chart rendering requested but LibreOffice/soffice is unavailable")

    converter = build_converter(
        render_office_charts=config.render_office_charts,
        images_scale=config.images_scale,
    )
    try:
        result = converter.convert(source_path)
    except Exception as exc:
        LOGGER.exception("Document conversion failed: %s", type(exc).__name__)
        raise
    doc = result.document
    page_count = len(_page_keys(doc))
    lookup = _item_lookup(doc)

    text_blocks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    image_hashes: dict[str, str] = {}
    table_hashes: dict[str, str] = {}
    order = 0
    last_page = 1
    # CHANGE 4: a page-sized image is only dropped when its OCR is already
    # present in that page's text layer, so the page text is collected first.
    page_text_by_page: dict[Any, list[str]] = {}
    # Text nested inside a picture is never exported on its own (the Markdown is
    # written with traverse_pictures=False), so for those words the picture is
    # the only carrier and must not be treated as a duplicate of page text.
    nested_in_picture = _refs_nested_in_pictures(doc, lookup)

    for item, level in _iter_items(doc):
        reported_page = _item_page(item, page_count)
        page = reported_page
        if page is None:
            page = last_page
        page_inferred = reported_page is None
        if reported_page is not None:
            last_page = reported_page
        source = _provenance(item)
        page_inferred = page_inferred or source.get("page") is None
        if isinstance(item, PictureItem):
            ocr_text = _descendant_text(item, lookup)
            coverage = _picture_coverage(doc, item)
            image, page_fallback, image_error = _image_from_item(
                doc, item, coverage, ocr_text
            )
            visual = image_signals.visual_signals(image)
            relative_path, image_hash, save_error, width, height = _save_image_asset(
                image,
                output_path / "images",
                "image",
                image_hashes,
            )
            del image
            if save_error:
                image_error = save_error
            image_record: dict[str, Any] = {
                "type": "image",
                "ref": getattr(item, "self_ref", None),
                "reference_id": getattr(item, "self_ref", None),
                "order": order,
                "level": level,
                "page": page,
                "page_inferred": page_inferred,
                "source": source,
                "path": relative_path,
                "relative_path": relative_path,
                "saved": relative_path is not None,
                "sha256": image_hash,
                "width": width,
                "height": height,
                "visual": visual,
                "position": image_signals.position_signals(
                    source.get("bbox"), _page_size(doc, page)
                ),
                "coverage": round(coverage, 6),
                "background": coverage >= BACKGROUND_COVERAGE,
                "page_fallback": page_fallback,
                "ocr": ocr_text,
                "ocr_characters": len(ocr_text),
                "include_in_md": True,
                "include_background": False,
                "include_decorative": config.include_decorative,
                "vision": None,
                "vision_status": None,
                "vision_error": None,
                **_picture_metadata(item),
            }
            if image_error:
                image_record["image_error"] = image_error
            image_record["include_background"] = image_record["background"]
            if config.include_background:
                image_record["include_in_md"] = True
            images.append(image_record)
            blocks.append(
                {
                    "type": "image",
                    "ref": image_record["ref"],
                    "order": order,
                    "page": page,
                    "level": level,
                }
            )
            order += 1
            continue

        if isinstance(item, TableItem):
            headers, rows = _table_rows(item, doc)
            try:
                table_markdown = item.export_to_markdown(doc=doc)
            except Exception as exc:
                table_markdown = ""
                LOGGER.warning("Table Markdown export failed: %s", type(exc).__name__)
            table_path, table_hash, table_error = _table_image(
                item,
                doc,
                output_path / "images",
                table_hashes,
                "table",
            )
            table_record: dict[str, Any] = {
                "type": "table",
                "ref": getattr(item, "self_ref", None),
                "reference_id": getattr(item, "self_ref", None),
                "order": order,
                "level": level,
                "page": page,
                "page_inferred": page_inferred,
                "source": source,
                "headers": headers,
                "rows": rows,
                "markdown": table_markdown,
                "complex": _table_is_complex(item),
                "image_path": table_path,
                "image_sha256": table_hash,
            }
            if table_error:
                table_record["image_error"] = table_error
            tables.append(table_record)
            blocks.append(
                {
                    "type": "table",
                    "ref": table_record["ref"],
                    "order": order,
                    "page": page,
                    "level": level,
                }
            )
            order += 1
            continue

        text = _as_text(getattr(item, "text", None))
        if text:
            ref = getattr(item, "self_ref", None)
            if ref not in nested_in_picture:
                page_text_by_page.setdefault(page, []).append(text)
            block = {
                "type": type(item).__name__,
                "ref": ref,
                "reference_id": ref,
                "order": order,
                "level": level,
                "page": page,
                "page_inferred": page_inferred,
                "source": source,
                "text": text,
            }
            text_blocks.append(block)
            blocks.append(
                {
                    "type": "text",
                    "ref": block["ref"],
                    "order": order,
                    "page": page,
                    "level": level,
                }
            )
            order += 1

    image_context = _classify_images(
        images,
        config=config,
        page_text=page_text_by_page,
        page_size_by_page={page: _page_size(doc, page) for page in _page_keys(doc)},
    )
    # CHANGE 5: identical image content is stored once, so all occurrences of an
    # asset must share one decision before anything is rendered or validated.
    asset_summary = (
        _link_duplicate_assets(images) if config.enable_asset_deduplication else {"unique_assets": len(images)}
    )
    LOGGER.info(
        "Asset dedup: %s unique asset(s), %s reused, %s duplicate occurrence(s)",
        asset_summary.get("unique_assets"),
        asset_summary.get("duplicate_assets", 0),
        asset_summary.get("duplicate_occurrences", 0),
    )
    LOGGER.info(
        "Image classification: %s meaningful, %s decorative, branding vocabulary=%s",
        sum(1 for record in images if record.get("include_in_md")),
        sum(1 for record in images if not record.get("include_in_md")),
        ",".join(image_context.get("branding_vocabulary") or []) or "none",
    )
    for record in images:
        if record.get("include_in_md"):
            continue
        LOGGER.info(
            "Omitted from Markdown: %s (page %s) class=%s evidence=%s",
            record.get("relative_path"),
            record.get("page"),
            record.get("semantic_class"),
            "; ".join(record.get("decorative_evidence") or []),
        )

    vision_stats = _apply_vision(images, output_path, config)
    decision_log = _write_image_decision_log(images, output_path)
    for record in images:
        LOGGER.info(
            "Image decision: %s page=%s class=%s confidence=%s (%s) via=%s kept=%s%s",
            record.get("relative_path") or record.get("ref"),
            record.get("page"),
            record.get("semantic_class"),
            record.get("classification_confidence"),
            record.get("classification_confidence_band"),
            record.get("classification_method"),
            record.get("include_in_md"),
            f" reason={record.get('omit_reason')}" if record.get("omit_reason") else "",
        )
    markdown = _render_document_markdown(
        doc,
        images,
        include_ocr=config.include_image_ocr,
        tables=tables,
    )
    if config.include_background:
        for record in images:
            if record.get("relative_path"):
                record["include_in_md"] = True
                record.pop("omit_reason", None)
        markdown = _render_document_markdown(
            doc,
            images,
            include_ocr=config.include_image_ocr,
            tables=tables,
        )
    markdown_path = output_path / "document.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    LOGGER.info(
        "Converted %s pages, %s text blocks, %s tables, %s images",
        page_count or 1,
        len(text_blocks),
        len(tables),
        len(images),
    )

    kept = [record for record in images if record.get("include_in_md")]
    omitted = [record for record in images if not record.get("include_in_md")]
    background_candidates = [record for record in images if record.get("background")]
    background_skipped = [
        record
        for record in omitted
        if record.get("background")
    ]
    semantic_classes = Counter(
        record.get("semantic_class") for record in images if record.get("semantic_class")
    )
    type_warnings = _type_aware_warnings(kept, config)
    page_stats = _page_stats(doc, text_blocks, tables, images)
    # The artifacts are written first and validation is folded in afterwards, so
    # that validation can check the files that were actually produced.
    validation: dict[str, Any] | None = None
    canonical: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "parser": PARSER_NAME,
        "source": source_meta,
        "source_path": str(source_path),
        "format": source_path.suffix.lower().lstrip("."),
        "created_at": started.isoformat(),
        "config": {
            "include_background": config.include_background,
            "include_decorative": config.include_decorative,
            "include_image_ocr": config.include_image_ocr,
            "render_office_charts": config.render_office_charts,
            "images_scale": config.images_scale,
            "vision_enabled": config.enable_vision and vision_available(),
            "vision_max_images": config.vision_max_images,
            "vision_timeout": config.vision_timeout,
        },
        "document": {
            "page_count": page_count or 1,
            "pages": page_stats,
        },
        "text": text_blocks,
        "tables": tables,
        "images": images,
        "blocks": blocks,
        "markdown": "document.md",
        "manifest": "manifest.json",
        "image_context": image_context,
        "stats": {
            "pages": page_count or 1,
            "text_blocks": len(text_blocks),
            "text_characters": sum(len(block.get("text", "")) for block in text_blocks),
            "tables": len(tables),
            "tables_complex": sum(1 for table in tables if table.get("complex")),
            "images_total": len(images),
            "images_saved": len({record.get("sha256") for record in images if record.get("sha256")}),
            "images_in_md": len(kept),
            "images_omitted": len(omitted),
            "images_meaningful": sum(1 for record in images if record.get("meaningful")),
            "images_decorative": sum(1 for record in images if not record.get("meaningful")),
            "unique_image_assets": asset_summary.get("unique_assets", len(images)),
            "duplicate_image_assets": asset_summary.get("duplicate_assets", 0),
            "duplicate_image_occurrences": asset_summary.get("duplicate_occurrences", 0),
            "images_text_layer_duplicates": sum(
                1
                for record in images
                if record.get("omit_reason") == "full_page_image_duplicates_text_layer"
            ),
            "image_classes": dict(sorted(semantic_classes.items())),
            "background_candidates": len(background_candidates),
            "background_skipped": len(background_skipped),
            "vision_enabled": bool(config.enable_vision and vision_available()),
            "vision_calls": vision_stats["calls"],
            "vision_api_calls": vision_stats.get("api_calls", vision_stats["calls"]),
            "vision_errors": vision_stats["errors"],
            "vision_retries": vision_stats.get("retried", 0),
            "vision_skipped": vision_stats["skipped"],
            "vision_confidence_threshold": vision_stats.get("confidence_threshold"),
            "vision_limit_reached": vision_stats.get("limit_reached", False),
            "vision_limit_skipped_highest_priority": vision_stats.get(
                "limit_skipped_highest_priority"
            ),
            "vision_statuses": vision_stats.get("statuses", {}),
            "vision_cost_usd": vision_stats["cost_usd"],
            "vision_tokens": vision_stats.get("tokens", {}),
            "validation_status": (validation or {}).get("status", "pending"),
            "semantic_loss_candidates": len((validation or {}).get("semantic_loss") or []),
            "type_aware_warnings": {
                WARNING_WARNING: sum(
                    1 for item in type_warnings if item["severity"] == WARNING_WARNING
                ),
                INFO_WARNING: sum(1 for item in type_warnings if item["severity"] == INFO_WARNING),
            },
        },
        "validation": validation if validation else {"status": "pending"},
    }
    # CHANGE 7/8: document.md and metadata.json are the deliverables; the large
    # canonical.json is opt-in. Both are written before validation runs so that
    # validation can check the real files, then rewritten with the verdict.
    canonical_path = output_path / "canonical.json"
    artifacts = {
        "markdown": "document.md",
        "metadata": "metadata.json" if config.emit_metadata else None,
        "canonical": "canonical.json" if config.emit_canonical else None,
        "manifest": "manifest.json",
    }
    if config.emit_canonical:
        _write_json(canonical_path, canonical)
    metadata: dict[str, Any] = {}
    if config.emit_metadata:
        metadata = _build_metadata(
            source_meta=source_meta,
            source_path=source_path,
            config=config,
            started=started,
            pages=page_stats,
            images=images,
            tables=tables,
            kept=kept,
            asset_summary=asset_summary,
            vision_stats=vision_stats,
            warnings=type_warnings,
            validation=None,
            artifacts=artifacts,
        )
        _write_json(output_path / "metadata.json", metadata)
    LOGGER.info("Wrote %s%s", markdown_path.name,
                " and metadata.json" if config.emit_metadata else "")

    if config.validate_output:
        validation = _validate_outputs(
            output_path, markdown, doc, images, tables, config=config
        )
    else:
        validation = {
            "status": "skipped",
            "errors": [],
            "warnings": [],
            "semantic_loss": [],
            "note": "validation was disabled for this run",
        }
        LOGGER.warning("Output validation was disabled (--no-validate)")
    canonical["validation"] = validation
    canonical["stats"]["validation_status"] = validation.get("status", "pending")
    if config.emit_canonical:
        _write_json(canonical_path, canonical)
    if config.emit_metadata and metadata:
        metadata["validation"] = dict(validation)
        _write_json(output_path / "metadata.json", metadata)

    # CHANGE 9: the manifest must describe the artifacts that were actually
    # written, not a fixed list, so a consumer can trust that every path it sees
    # here exists and that nothing exists without being listed.
    manifest = {
        "parser": "docling-slim",
        "schema_version": SCHEMA_VERSION,
        "source": source_meta,
        "created_at": started.isoformat(),
        "completed_at": None,
        "status": "in_progress",
        "validation": {
            "status": validation["status"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        },
        "emitted": dict(artifacts),
        "artifacts": [],
        "statistics": canonical["stats"],
        "pages": page_count or 1,
        "notes": "manifest.json is written last and is therefore not self-checksummed",
    }
    written = [markdown_path]
    metadata_path = output_path / "metadata.json"
    if config.emit_metadata and metadata_path.is_file():
        written.append(metadata_path)
    if config.emit_canonical and canonical_path.is_file():
        written.append(canonical_path)
    for log_artifact in (output_path / "logs" / "parser.log", decision_log):
        if log_artifact and log_artifact.is_file():
            written.append(log_artifact)
    written.extend(sorted((output_path / "images").glob("*.png")))
    for artifact in written:
        manifest["artifacts"].append(
            {
                "path": artifact.relative_to(output_path).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
            }
        )
    manifest["artifact_counts"] = {
        "total": len(manifest["artifacts"]),
        "images": sum(1 for item in manifest["artifacts"] if item["path"].startswith("images/")),
        "tables": sum(1 for item in manifest["artifacts"] if item["path"].startswith("images/table_")),
    }
    integrity = _verify_manifest_contents(output_path, manifest)
    if integrity["errors"]:
        for message in integrity["errors"]:
            LOGGER.error("Manifest integrity: %s", message)
            validation["errors"].append(f"Manifest integrity: {message}")
        if validation["status"] == "pass":
            validation["status"] = "warning"
        manifest["validation"] = {
            "status": validation["status"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }
    manifest["integrity"] = {
        "checked": integrity["checked"],
        "errors": integrity["errors"],
        "unlisted_files": integrity["unlisted_files"],
    }
    manifest["status"] = "complete" if validation["status"] != "fail" else "failed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_path / "manifest.json", manifest)
    LOGGER.info(
        "Wrote %s with validation status %s (%s images kept, %s omitted)",
        markdown_path,
        validation["status"],
        len(kept),
        len(omitted),
    )
    if validation["status"] == "fail":
        LOGGER.error("Validation failed: %s", "; ".join(validation["errors"]))
    elif validation["status"] == "warning":
        LOGGER.warning(
            "Potential semantic loss detected: %s",
            "; ".join(item["message"] for item in validation.get("semantic_loss") or []),
        )
    return canonical


def _verify_manifest_contents(output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Check the manifest against what is really on disk (CHANGE 9/10).

    Two directions matter: every listed artifact must exist with the recorded
    checksum, and every file that was produced must be listed. The first catches
    a truncated or rewritten file, the second catches an artifact that a consumer
    would never learn about.
    """
    errors: list[str] = []
    listed: set[str] = set()
    for artifact in manifest.get("artifacts") or []:
        relative = _as_text(artifact.get("path"))
        listed.add(relative)
        target = output_dir / relative
        if not target.is_file():
            errors.append(f"listed artifact does not exist: {relative}")
            continue
        expected = _as_text(artifact.get("sha256"))
        if expected and _sha256_file(target) != expected:
            errors.append(f"checksum mismatch: {relative}")
    unlisted: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative == "manifest.json" or relative in listed:
            continue
        unlisted.append(relative)
    return {
        "checked": len(listed),
        "errors": errors,
        "unlisted_files": unlisted,
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse PDF, DOCX, and PPTX documents with Docling")
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help=(
            "One or more PDF/DOCX/PPTX files, directories to walk, or .zip archives. "
            "Several documents each get their own subdirectory under --output."
        ),
    )
    parser.add_argument("--output", default="output_docling")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--include-decorative", action="store_true")
    parser.add_argument("--no-image-ocr", action="store_true")
    parser.add_argument(
        "--render-office-charts",
        action="store_true",
        help="Render native DOCX/PPTX charts when LibreOffice is available",
    )
    vision_group = parser.add_mutually_exclusive_group()
    vision_group.add_argument(
        "--vision",
        dest="enable_vision",
        action="store_true",
        help="Enable the optional Vision LLM integration",
    )
    vision_group.add_argument(
        "--no-vision",
        dest="enable_vision",
        action="store_false",
        help="Disable Vision processing (default)",
    )
    parser.set_defaults(enable_vision=False)
    parser.add_argument("--vision-max-images", type=int, default=None)
    parser.add_argument("--vision-timeout", type=float, default=60.0)
    parser.add_argument("--vision-retry-count", type=int, default=1)
    parser.add_argument(
        "--vision-confidence-threshold",
        type=float,
        default=DEFAULT_VISION_CONFIDENCE_THRESHOLD,
        help="Send an image to Vision when its deterministic confidence is below this value",
    )
    parser.add_argument(
        "--no-asset-deduplication",
        action="store_true",
        help="Store every extracted asset separately, even when the bytes are identical",
    )
    parser.add_argument(
        "--no-full-page-duplicate-detection",
        action="store_true",
        help="Keep page-sized images even when they only duplicate the page text layer",
    )
    canonical_group = parser.add_mutually_exclusive_group()
    canonical_group.add_argument(
        "--emit-canonical",
        dest="emit_canonical",
        action="store_true",
        help="Also write the full canonical.json (debug/audit/reprocessing)",
    )
    canonical_group.add_argument(
        "--no-canonical",
        dest="emit_canonical",
        action="store_false",
        help="Skip canonical.json (default: only document.md, metadata.json, manifest.json)",
    )
    parser.set_defaults(emit_canonical=False)
    parser.add_argument(
        "--no-metadata",
        dest="emit_metadata",
        action="store_false",
        help="Skip metadata.json",
    )
    parser.set_defaults(emit_metadata=True)
    parser.add_argument(
        "--no-validate",
        dest="validate_output",
        action="store_false",
        help="Skip output validation (not recommended)",
    )
    parser.set_defaults(validate_output=True)
    parser.add_argument(
        "--images-scale",
        type=float,
        default=2.0,
        help="Scale factor for extracted image/OCR assets (lower values use less memory)",
    )

    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    options = dict(
        include_background=args.include_background,
        include_decorative=args.include_decorative,
        include_image_ocr=not args.no_image_ocr,
        render_office_charts=args.render_office_charts,
        enable_vision=args.enable_vision,
        vision_max_images=args.vision_max_images,
        vision_timeout=args.vision_timeout,
        vision_retry_count=args.vision_retry_count,
        vision_confidence_threshold=args.vision_confidence_threshold,
        enable_asset_deduplication=not args.no_asset_deduplication,
        enable_full_page_duplicate_detection=not args.no_full_page_duplicate_detection,
        emit_canonical=args.emit_canonical,
        emit_metadata=args.emit_metadata,
        validate_output=args.validate_output,
        images_scale=args.images_scale,
        force=args.force,
        log_level=args.log_level,
    )
    # A lone, already-supported file keeps the original single-document code path,
    # so existing invocations are unaffected. Anything else -- several inputs, a
    # directory, or a zip -- goes through parse_many, which resolves it first.
    is_single_document = (
        len(args.inputs) == 1 and Path(args.inputs[0]).suffix.lower() in SUPPORTED
    )
    if is_single_document:
        result = parse_document(args.inputs[0], args.output, **options)
        print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
        if result["validation"]["status"] == "fail":
            raise SystemExit(2)
        if result["validation"]["status"] == "warning":
            raise SystemExit(0)
        return

    summary = parse_many(args.inputs, args.output, **options)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed"]:
        raise SystemExit(2)
    if any(entry.get("validation") == "fail" for entry in summary["results"]):
        raise SystemExit(2)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
