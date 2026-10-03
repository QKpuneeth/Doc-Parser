"""Deterministic, multi-signal classification of extracted images.

The goal of this module is to answer one question per image:

    does this image carry information that a downstream pipeline (chunking,
    embeddings, retrieval) would lose if the image were dropped?

Image geometry (pixel area, page coverage) and OCR length are deliberately
never used on their own: a 248x291 pixel pie chart holding ``14% / 19% / 35%``
is far more valuable than a 1600x400 pixel gradient banner.

Signals are grouped in three families:

* ``text``     - derived from OCR text (percentages, numbers, labels, domain
                 vocabulary, table-like layout, garbling, branding, page numbers)
* ``visual``   - derived from the rasterised asset (dimensions, colour
                 complexity, ink coverage) and used only as supporting evidence
* ``context``  - derived from the document as a whole (repeated brand elements
                 in the same page region, duplicated assets)

``classify_image`` combines all of them and returns the decision together with
the evidence that produced it, so canonical.json can explain every decision.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

BACKGROUND_COVERAGE = 0.6

_TINY_DIMENSION = 24
_TINY_AREA = 2_500
_THIN_BAND_RATIO = 2.5
_FLAT_COLOR_COUNT = 3
_BLANK_INK_RATIO = 0.02
_BRAND_IMAGE_MAX_COVERAGE = 0.2
_BRAND_IMAGE_MAX_AREA = 400_000
_BRAND_TEXT_MAX_TOKENS = 4
_SMALL_IMAGE_MAX_COVERAGE = 0.08
_PAGE_SCALE_COVERAGE = 0.4
_SUBSTANTIAL_OCR_CHARACTERS = 40
_SUBSTANTIAL_OCR_WORDS = 8
_STRONG_NUMBER_COUNT = 2

CHART_TERMS = frozenset(
    {
        "axis", "axes", "bar", "bars", "bargraph", "barchart", "boxplot",
        "bubble", "curve", "curvechart", "donut", "funnel", "gauge", "graph",
        "histogram", "line", "linechart", "pie", "piechart", "plot", "radar",
        "scatter", "sparkline", "stacked", "trend", "yaxis", "xaxis",
    }
)

DIAGRAM_TERMS = frozenset(
    {
        "arrow", "arrows", "cycle", "diagram", "flow", "flowchart", "framework",
        "hierarchy", "matrix", "mindmap", "network", "orgchart", "process",
        "quadrant", "roadmap", "schematic", "stage", "steps", "swimlane",
        "timeline", "topology", "workflow",
    }
)

CHART_PHRASES = (
    "% of",
    "per cent",
    "percent",
    "percentage",
    "distribution",
    "share of",
    "market share",
    "by segment",
    "year over year",
    "yoy",
    "quarter over quarter",
    "qoq",
    "total (n=",
    "n=",
    "axis",
    "legend",
    "categories",
)

TABLE_TERMS = frozenset(
    {
        "category", "categories", "col", "column", "measure", "metric", "row",
        "segment", "table", "total", "value", "values",
    }
)

DOMAIN_TERMS = frozenset(
    {
        # medical / life sciences
        "adverse", "biomarker", "cardiac", "clinical", "cohort", "diagnosis",
        "disease", "dose", "dosing", "efficacy", "episode", "guideline", "hg",
        "hospital", "incidence", "indication", "infusion", "inpatient", "lab",
        "lesion", "medication", "mg", "ml", "mmhg", "molecule", "mortality",
        "onset", "patient", "patients", "physician", "physicians", "placebo",
        "prevalence", "prescriber", "prescribers", "proportion", "psychiatrist",
        "psychiatrists", "randomised", "randomized", "regimen", "remission",
        "respondent", "respondents", "response", "safety", "screening",
        "symptom", "symptoms", "tolerance", "treatment", "trial", "tumor",
        "tumour", "vaccine",
        # scientific
        "analysis", "coefficient", "confidence", "correlation", "estimate",
        "hypothesis", "interval", "median", "p-value", "probability", "regression",
        "sample", "sample-size", "significant", "standard", "variance",
        # business / finance
        "accounts", "annual", "budget", "cost", "costs", "customer", "ebitda",
        "forecast", "growth", "margin", "market", "pricing", "profit",
        "quarterly", "revenue", "roi", "sales", "share", "target", "territory",
    }
)

STOPWORDS = frozenset(
    {
        "about", "after", "again", "against", "all", "also", "among", "and",
        "any", "are", "because", "been", "before", "being", "below", "between",
        "both", "but", "can", "could", "did", "does", "doing", "done", "down",
        "during", "each", "few", "for", "from", "further", "had", "has",
        "have", "having", "here", "how", "into", "its", "itself", "just",
        "let", "like", "may", "might", "more", "most", "much", "must", "not",
        "now", "off", "once", "only", "other", "our", "out", "over", "own",
        "same", "shall", "should", "some", "such", "than", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "through", "too", "under", "until", "very", "was", "were", "what",
        "when", "where", "which", "while", "who", "why", "will", "with",
        "within", "would", "you", "your",
    }
)

GENERIC_BRAND_TERMS = frozenset(
    {
        "confidential", "copyright", "draft", "logo", "sample", "template",
        "trademark", "watermark", "reserved", "rights", "inc", "ltd", "llc",
        "corp", "corporation", "gmbh", "plc", "company", "limited", "holdings",
        "registered", "unauthorized", "do", "not", "print",
    }
)

PERCENT_RE = re.compile(r"[~≈<>^+-]?\s?\d{1,3}(?:[.,]\d+)?\s*(?:%|percent\b|pct\b)")
CURRENCY_RE = re.compile(r"(?:[$€£¥₹]\s?\d)|(?:\b(?:USD|EUR|GBP|INR|JPY)\b)")
NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?![\w])")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-/]{2,}")
UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|ml|µg|ug|mmhg|kg|cm|mm|hz|khz|mhz|ghz|"
    r"bps|°c|°f|celsius|years?|months?|days?|weeks?|hours?|minutes?)\b",
    re.IGNORECASE,
)
PAGE_NUMBER_RE = re.compile(
    r"^(?:\s*(?:page|pg|p|slide|s)?\s*[-–—]?\s*\d{1,4}"
    r"(?:\s*(?:of|/|-|–)\s*\d{1,4})?|\d{1,4}\s*[-–—|]\s*\d{1,4})\s*\.?$",
    re.IGNORECASE,
)
ASCII_JUNK_RE = re.compile(r"[^\x20-\x7E\n\t]")

SEMANTIC_CHART = "chart"
SEMANTIC_DIAGRAM = "diagram"
SEMANTIC_TABLE = "table_image"
SEMANTIC_TEXT = "text_image"
SEMANTIC_PHOTO = "photo"
SEMANTIC_LOGO = "logo"
SEMANTIC_PAGE_NUMBER = "page_number"
SEMANTIC_DECORATION = "icon_or_decoration"
SEMANTIC_BACKGROUND = "background"
SEMANTIC_UNCLASSIFIED = "unclassified_image"
SEMANTIC_DUPLICATE = "duplicate"

# Classes that a reader would expect to find represented in the extracted
# content, as opposed to page furniture (branding, icons, page numbers).
CONTENT_IMAGE_CLASSES = frozenset(
    {SEMANTIC_CHART, SEMANTIC_DIAGRAM, SEMANTIC_TABLE, SEMANTIC_TEXT, SEMANTIC_PHOTO}
)
# Classes that are page furniture rather than document content.
FURNITURE_IMAGE_CLASSES = frozenset(
    {SEMANTIC_LOGO, SEMANTIC_PAGE_NUMBER, SEMANTIC_DECORATION, SEMANTIC_BACKGROUND, SEMANTIC_DUPLICATE}
)

VISION_MEANINGFUL_CATEGORIES = frozenset({"diagram_chart", "text_image"})

# --- classification confidence (CHANGE 2) ------------------------------------
# Confidence is derived from the weight of the evidence that produced the
# verdict. It is deliberately explainable rather than calibrated: the ledger of
# contributing signals is returned alongside the number so every decision can be
# audited. Bands follow the documented high/medium/low vocabulary.
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.45
# The band labels themselves, so callers never hard-code "high"/"medium"/"low".
BAND_HIGH = "high"
BAND_MEDIUM = "medium"
BAND_LOW = "low"
CONFIDENCE_BANDS = (BAND_HIGH, BAND_MEDIUM, BAND_LOW)
CONFIDENCE_SATURATION = 2.5
CONFIDENCE_CONFLICT_PENALTY = 0.2
METHOD_DETERMINISTIC = "deterministic"
METHOD_DOCLING = "docling"
METHOD_VISION = "vision"
METHOD_DEFAULT = "default"

# Evidence weights, keyed by the signal that produced them. Strong signals
# (structured chart metadata, percentages, several distinct numbers) are worth
# more than vocabulary hints, which are worth more than size/position hints.
EVIDENCE_WEIGHTS = {
    "chart_data": 1.0,
    "chart_annotation": 1.0,
    "docling_chart_class": 0.8,
    "percentages": 0.9,
    "numeric_values": 0.6,
    "currency": 0.7,
    "units": 0.6,
    "chart_vocabulary": 0.6,
    "diagram_vocabulary": 0.6,
    "table_structure": 0.6,
    "domain_vocabulary": 0.4,
    "text_labels": 0.4,
    "substantial_text": 0.4,
    "vision_classification": 0.9,
    "vision_description": 0.8,
    "page_scale_text": 0.3,
    "branding_only": 0.9,
    "branding_repeated": 0.8,
    "page_number": 0.9,
    "repeated_asset": 0.6,
    "empty_asset": 0.8,
    "icon_sized": 0.7,
    "page_background": 0.6,
    "missing_asset": 1.0,
    "text_layer_duplicate": 1.0,
}

# Signals that point in opposite directions and therefore reduce confidence.
MEANINGFUL_STRONG_SIGNALS = frozenset(
    {
        "chart_data",
        "chart_annotation",
        "percentages",
        "numeric_values",
        "currency",
        "units",
        "chart_vocabulary",
        "diagram_vocabulary",
        "table_structure",
        "substantial_text",
        "vision_classification",
        "vision_description",
    }
)
DECORATIVE_STRONG_SIGNALS = frozenset(
    {
        "branding_only",
        "branding_repeated",
        "page_number",
        "empty_asset",
        "icon_sized",
        "page_background",
        "missing_asset",
        "text_layer_duplicate",
    }
)
CONFIDENCE_BASE_WEIGHT = 0.7
CONFIDENCE_CORROBORATION_WEIGHT = 0.3
CONFIDENCE_CORROBORATION_SCALE = 1.0
CONFIDENCE_CONFLICT_PENALTY = 0.25
CONFIDENCE_MAX_CONFLICTS = 2

# --- full page duplicate detection (CHANGE 4) -------------------------------
MIN_DUPLICATE_TOKENS = 12


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[\s,;:()\[\]{}<>|/\\]+", text or "") if token]


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _squash(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def is_brand_token(token: str, vocabulary: Iterable[str] = ()) -> bool:
    """Heuristically decide whether a token looks like a brand/company name."""
    squashed = _squash(token)
    if not squashed:
        return False
    vocabulary = {squashed for squashed in (_squash(item) for item in vocabulary) if squashed}
    if squashed in vocabulary:
        return True
    if squashed in GENERIC_BRAND_TERMS:
        return True
    for candidate in vocabulary | GENERIC_BRAND_TERMS:
        if abs(len(candidate) - len(squashed)) > 3:
            continue
        if difflib.SequenceMatcher(None, squashed, candidate).ratio() >= 0.78:
            return True
    return token.isupper() and 2 <= len(squashed) <= 12 and squashed.isalnum()


def text_signals(text: str, brand_vocabulary: Iterable[str] = ()) -> dict[str, Any]:
    """Derive content signals from OCR text."""
    raw = text or ""
    stripped = raw.strip()
    tokens = _tokens(raw)
    words = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    lower_words = [token.lower() for token in words]
    percentages = [match.group(0).strip() for match in PERCENT_RE.finditer(raw)]
    numbers = [match.group(0).strip() for match in NUMBER_RE.finditer(raw)]
    content_words = [
        WORD_RE.match(token).group(0).lower()
        for token in words
        if token not in STOPWORDS and WORD_RE.match(token)
    ]
    brand_vocabulary_list = list(brand_vocabulary)
    brand_like_tokens = [
        token for token in words if is_brand_token(token, brand_vocabulary_list)
    ]
    content_words = [
        word for word in content_words
        if WORD_RE.match(word).group(0) not in brand_like_tokens
        and word not in {token.lower() for token in brand_like_tokens}
    ]
    chart_terms = sorted(
        {
            word
            for word in lower_words
            for term in CHART_TERMS
            if term in word
        }
    )
    diagram_terms = sorted(
        {
            word
            for word in lower_words
            for term in DIAGRAM_TERMS
            if term in word
        }
    )
    table_terms = sorted(
        {
            word
            for word in lower_words
            for term in TABLE_TERMS
            if term in word
        }
    )
    domain_terms = sorted(
        {
            word
            for word in lower_words
            for term in DOMAIN_TERMS
            if term in word
        }
    )
    phrases = [phrase for phrase in CHART_PHRASES if phrase in raw.lower()]

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    numeric_lines = [
        line
        for line in lines
        if line and NUMBER_RE.search(line) and len(_tokens(line)) <= 3
    ]
    multi_token_lines = [line for line in lines if len(_tokens(line)) >= 2]
    table_like = False
    if len(lines) >= 3:
        grid_ratio = len(multi_token_lines) / len(lines)
        numeric_ratio = len(numeric_lines) / len(lines)
        if grid_ratio >= 0.6 and all(len(line) <= 40 for line in lines):
            table_like = True
        if numeric_ratio >= 0.5 and len(numeric_lines) >= 3:
            table_like = True
    if "|" in raw and raw.count("|") >= 2:
        table_like = True

    single_char_ratio = (
        sum(1 for token in tokens if len(token.strip(".,")) == 1) / len(tokens)
        if tokens
        else 0.0
    )
    alnum = sum(1 for char in raw if char.isalnum())
    replacement_chars = raw.count("�") + raw.count("□")
    alnum_ratio = alnum / len(raw.replace("\n", "")) if raw.strip() else 1.0
    garbled = bool(replacement_chars) or (
        len(tokens) >= 4 and single_char_ratio >= 0.4
    ) or (
        len(raw) >= 20 and alnum_ratio < 0.55
    )

    brand_vocabulary_list = list(brand_vocabulary)
    brand_like_tokens = [
        token for token in words if is_brand_token(token, brand_vocabulary_list)
    ]
    non_brand_words = [token for token in words if token not in brand_like_tokens]
    brand_only = bool(words) and not non_brand_words and len(words) <= _BRAND_TEXT_MAX_TOKENS
    return {
        "characters": len(stripped),
        "words": len(words),
        "lines": len(lines),
        "tokens": len(tokens),
        "empty": not stripped,
        "has_percent": bool(percentages),
        "percent_values": percentages,
        "has_currency": bool(CURRENCY_RE.search(raw)),
        "has_units": bool(UNIT_RE.search(raw)),
        "numeric_values": numbers,
        "numeric_count": len(numbers),
        "distinct_numeric_count": len(set(numbers)),
        "content_words": content_words,
        "content_word_count": len(content_words),
        "chart_terms": sorted(set(chart_terms)),
        "chart_phrases": phrases,
        "diagram_terms": diagram_terms,
        "table_terms": table_terms,
        "table_like": table_like,
        "domain_terms": domain_terms,
        "garbled": garbled,
        "single_char_ratio": round(single_char_ratio, 4),
        "brand_tokens": brand_like_tokens,
        "brand_only": brand_only,
        "page_number_only": bool(stripped) and not brand_only and bool(PAGE_NUMBER_RE.match(stripped)),
        "has_non_ascii_junk": bool(ASCII_JUNK_RE.search(raw)),
    }


def visual_signals(image: Any) -> dict[str, Any]:
    """Cheap, deterministic raster statistics (never decisive on their own)."""
    result: dict[str, Any] = {
        "width": None,
        "height": None,
        "area": 0,
        "aspect_ratio": None,
        "min_dimension": 0,
        "tiny": False,
        "thin_band": False,
        "flat": False,
        "near_blank": False,
        "distinct_colors": None,
        "ink_ratio": None,
    }
    if image is None:
        return result
    try:
        width, height = image.size
    except Exception:
        return result
    result["width"] = int(width)
    result["height"] = int(height)
    result["area"] = int(width) * int(height)
    if height:
        result["aspect_ratio"] = round(float(width) / float(height), 4)
    result["min_dimension"] = int(min(width, height))
    result["tiny"] = result["min_dimension"] <= _TINY_DIMENSION or result["area"] <= _TINY_AREA
    result["thin_band"] = bool(
        height and width and (float(width) / float(height) >= _THIN_BAND_RATIO)
    )
    try:
        sample = image.convert("RGB").resize((64, 64))
        colors = sample.getcolors(64 * 64) or []
    except Exception:
        return result
    if not colors:
        return result
    colors.sort(reverse=True)
    result["distinct_colors"] = len(colors)
    result["flat"] = len(colors) <= _FLAT_COLOR_COUNT
    background = colors[0][1]
    total = 64 * 64
    inked = sum(
        count
        for count, color in colors
        if max(abs(color[index] - background[index]) for index in range(3)) > 18
    )
    ink_ratio = inked / total
    result["ink_ratio"] = round(ink_ratio, 4)
    result["near_blank"] = ink_ratio <= _BLANK_INK_RATIO
    return result


def position_signals(
    bbox: Sequence[float] | None,
    page_size: tuple[float, float] | None,
) -> dict[str, Any]:
    """Relative placement of an image, used for repeated-branding detection."""
    result: dict[str, Any] = {
        "left_ratio": None,
        "top_ratio": None,
        "width_ratio": None,
        "height_ratio": None,
        "band": None,
    }
    if not bbox or len(bbox) < 4 or not page_size:
        return result
    page_width, page_height = page_size
    if not page_width or not page_height:
        return result
    left, top, right, bottom = (float(value) for value in bbox[:4])
    left, right = sorted((max(0.0, left), max(0.0, right)))
    top, bottom = sorted((max(0.0, top), max(0.0, bottom)))
    left_ratio = left / page_width
    top_ratio = top / page_height
    width_ratio = (min(right, page_width) - left) / page_width
    height_ratio = (min(bottom, page_height) - top) / page_height
    vertical = "top" if top_ratio < 0.2 else ("bottom" if top_ratio > 0.8 else "middle")
    horizontal = "left" if left_ratio < 0.33 else ("right" if left_ratio > 0.66 else "center")
    result.update(
        {
            "left_ratio": round(left_ratio, 4),
            "top_ratio": round(top_ratio, 4),
            "width_ratio": round(width_ratio, 4),
            "height_ratio": round(height_ratio, 4),
            "band": f"{vertical}-{horizontal}",
        }
    )
    return result


def band_key(record: Mapping[str, Any]) -> str | None:
    position = record.get("position") or {}
    band = position.get("band")
    return str(band) if band else None


def build_document_context(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Learn document-level evidence: brand vocabulary and repeated page furniture.

    Branding is discovered from the document itself rather than hard-coded: short
    text found in the same page region on several different pages is treated as
    branding vocabulary. Exact-duplicate assets across pages are also collected.
    """
    band_pages: dict[str, set[Any]] = {}
    band_candidates: dict[str, list[str]] = {}
    hash_pages: dict[str, set[Any]] = {}
    for record in records:
        band = band_key(record)
        if band:
            band_pages.setdefault(band, set()).add(record.get("page"))
        text = record.get("ocr") or ""
        coverage = float(record.get("coverage") or 0.0)
        tokens = _tokens(text)
        if band and coverage <= _SMALL_IMAGE_MAX_COVERAGE and 0 < len(tokens) <= 3:
            for token in tokens:
                if WORD_RE.match(token):
                    band_candidates.setdefault(band, []).append(token)
        digest = record.get("sha256")
        if digest:
            hash_pages.setdefault(str(digest), set()).add(record.get("page"))

    repeated_bands = {
        band
        for band, pages in band_pages.items()
        if len({page for page in pages if page is not None}) >= 2
    }
    vocabulary: Counter[str] = Counter()
    for band, candidates in band_candidates.items():
        if band not in repeated_bands:
            continue
        for token in candidates:
            if is_brand_token(token):
                vocabulary[_squash(token)] += 1

    repeated_assets = {
        digest: sorted(str(page) for page in pages if page is not None)
        for digest, pages in hash_pages.items()
        if len({page for page in pages if page is not None}) >= 2
    }
    return {
        "branding_vocabulary": sorted(vocabulary),
        "repeated_bands": sorted(repeated_bands),
        "repeated_assets": repeated_assets,
        "record_count": len(records),
    }


def _confidence_band(value: float) -> str:
    if value >= CONFIDENCE_HIGH:
        return BAND_HIGH
    if value >= CONFIDENCE_MEDIUM:
        return BAND_MEDIUM
    return BAND_LOW


def score_confidence(
    ledger: Sequence[Mapping[str, Any]],
    *,
    method: str = METHOD_DETERMINISTIC,
) -> tuple[float, str, str]:
    """Turn a weighted evidence ledger into ``(confidence, band, method)``.

    The strongest single signal sets the base of the score and independent
    corroborating signals refine it, so one decisive observation (page number,
    structured chart metadata) already scores higher than a pile of weak hints,
    while genuinely contradictory evidence is penalised. The ledger travels with
    the verdict so the number is always explainable.
    """
    strengths: dict[str, float] = {}
    for item in ledger:
        signal = str(item.get("signal") or "")
        if not signal:
            continue
        try:
            weight = float(item.get("weight") or 0.0)
        except (TypeError, ValueError):
            continue
        strengths[signal] = max(strengths.get(signal, 0.0), weight)

    if not strengths:
        confidence = 0.0
    else:
        base = max(strengths.values())
        corroboration = min(1.0, (sum(strengths.values()) - base) / CONFIDENCE_CORROBORATION_SCALE)
        confidence = (
            CONFIDENCE_BASE_WEIGHT * base + CONFIDENCE_CORROBORATION_WEIGHT * corroboration
        )

    supports_meaningful = bool(strengths.keys() & MEANINGFUL_STRONG_SIGNALS)
    supports_decorative = bool(strengths.keys() & DECORATIVE_STRONG_SIGNALS)
    conflicts = (
        min(
            len(strengths.keys() & MEANINGFUL_STRONG_SIGNALS),
            len(strengths.keys() & DECORATIVE_STRONG_SIGNALS),
            CONFIDENCE_MAX_CONFLICTS,
        )
        if supports_meaningful and supports_decorative
        else 0
    )
    if method == METHOD_DEFAULT:
        conflicts += 1
    confidence = round(max(0.0, min(1.0, confidence * (1.0 - CONFIDENCE_CONFLICT_PENALTY * conflicts))), 2)
    return confidence, _confidence_band(confidence), method


def _comparison_tokens(text: str) -> set[str]:
    """Set of comparable tokens: words and numeric values, lower-cased.

    Presence rather than multiplicity is what matters: a scanned page usually
    OCRs the same sentence once while the text layer stores it once, and a
    figure may repeat a caption. Numeric values are kept as tokens too, so a
    chart whose numbers are already in the text layer is recognised as a
    duplicate while one introducing new numbers is not.
    """
    lowered = (text or "").lower()
    words = {_squash(token) for token in _tokens(lowered) if WORD_RE.match(token)}
    numbers = {
        match.group(0).replace(",", "").replace(" ", "")
        for match in NUMBER_RE.finditer(lowered)
    }
    return {token for token in words | numbers if token}


def text_duplicate_signals(
    image_text: str,
    page_text: str,
    *,
    similarity_threshold: float = 0.9,
    min_characters: int = 120,
) -> dict[str, Any]:
    """Compare image OCR against an existing text layer.

    Used for page-sized images: if the OCR of such an image is already fully
    present in the page's text layer, keeping both would double the semantic
    content of the document. The comparison is deliberately conservative -
    anything short, partially overlapping or numerically different is reported
    as *not* a duplicate so unique information is never dropped.
    """
    image_raw = (image_text or "").strip()
    page_raw = (page_text or "").strip()
    image_tokens = _comparison_tokens(image_raw)
    page_tokens = _comparison_tokens(page_raw)
    shared = image_tokens & page_tokens
    token_coverage = len(shared) / len(image_tokens) if image_tokens else 0.0
    # Character-weighted coverage answers the question that actually matters:
    # how much of this image's text does the page already carry? A set of
    # distinct words saturates quickly and would hide a short but unique
    # caption inside an otherwise duplicated page.
    characters = len(re.sub(r"\s+", "", image_raw))
    unique_characters = sum(len(_squash(token)) for token in image_tokens)
    character_coverage = (
        sum(len(_squash(token)) for token in shared) / unique_characters if unique_characters else 0.0
    )
    duplicate = bool(
        characters >= int(min_characters)
        and len(image_tokens) >= MIN_DUPLICATE_TOKENS
        and token_coverage >= float(similarity_threshold)
        and character_coverage >= float(similarity_threshold)
    )
    return {
        "duplicate": duplicate,
        "token_coverage": round(token_coverage, 4),
        "character_coverage": round(character_coverage, 4),
        "unique_image_tokens": len(image_tokens),
        "unique_page_tokens": len(page_tokens),
        "compared_characters": characters,
        "min_characters": int(min_characters),
        "similarity_threshold": float(similarity_threshold),
    }



def _classification_terms(record: Mapping[str, Any]) -> str:
    values = [str(record.get("classification") or ""), str(record.get("semantic_class") or "")]
    meta = record.get("picture_metadata") or {}
    annotations = meta.get("annotations")
    if isinstance(annotations, list):
        values.extend(str(item) for item in annotations)
    return " ".join(value.lower() for value in values if value)


def _docling_class(record: Mapping[str, Any]) -> str | None:
    meta = record.get("picture_metadata") or {}
    classification = meta.get("classification")
    if classification:
        return str(classification).strip().lower() or None
    value = record.get("classification")
    text = str(value).strip().lower() if value else ""
    if not text or text in {SEMANTIC_UNCLASSIFIED, "other"}:
        return None
    return text


def _chart_annotation(record: Mapping[str, Any]) -> bool:
    meta = record.get("picture_metadata") or {}
    annotations = meta.get("annotations") or []
    chart_kinds = {
        "picture_tabular_chart_data",
        "picture_line_chart_data",
        "picture_bar_chart_data",
        "picture_stacked_bar_chart_data",
        "picture_pie_chart_data",
        "picture_scatter_chart_data",
    }
    return any(str(kind) in chart_kinds for kind in annotations)


def _vision(record: Mapping[str, Any]) -> Mapping[str, Any]:
    vision = record.get("vision")
    return vision if isinstance(vision, Mapping) else {}


def classify_image(
    record: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    image: Any = None,
    *,
    full_page_duplicate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one image record into a semantic class + Markdown inclusion verdict.

    Returns a dictionary with ``meaningful``, ``semantic_class``, the evidence
    lists, and every signal used, so the decision is fully auditable.
    """
    context = context or {}
    vocabulary = list(context.get("branding_vocabulary") or [])
    repeated_bands = set(context.get("repeated_bands") or ())
    repeated_assets = context.get("repeated_assets") or {}

    text = record.get("ocr") or ""
    coverage = float(record.get("coverage") or 0.0)
    background = bool(record.get("background")) or coverage >= BACKGROUND_COVERAGE
    has_chart_data = bool(record.get("chart_data"))
    image_path = record.get("relative_path") or record.get("path")
    saved = bool(image_path)

    precomputed_visual = record.get("visual")
    if isinstance(precomputed_visual, Mapping) and precomputed_visual:
        visual = dict(precomputed_visual)
    elif image is not None:
        visual = visual_signals(image)
    else:
        visual = {
            "width": record.get("width"),
            "height": record.get("height"),
            "area": int(record.get("width") or 0) * int(record.get("height") or 0),
        }

    signals = {
        "text": text_signals(text, vocabulary),
        "visual": visual,
        "position": dict(record.get("position") or {}),
    }
    text_signal = signals["text"]
    position = signals["position"]
    band = position.get("band")
    if record.get("width") and record.get("height") and not visual.get("area"):
        visual["area"] = int(record["width"]) * int(record["height"])
        visual["min_dimension"] = min(int(record["width"]), int(record["height"]))
        visual["aspect_ratio"] = round(
            float(record["width"]) / float(record["height"]), 4
        ) if record["height"] else None
        visual["tiny"] = visual["min_dimension"] <= _TINY_DIMENSION or visual["area"] <= _TINY_AREA
        visual["thin_band"] = bool(visual.get("aspect_ratio") and visual["aspect_ratio"] >= _THIN_BAND_RATIO)

    docling_class = _docling_class(record)
    classification_terms = _classification_terms(record)
    vision = _vision(record)
    vision_status = str(record.get("vision_status") or "")
    vision_category = str(vision.get("category") or "").strip().lower()
    vision_description = str(vision.get("description") or "").strip()
    vision_text = str(vision.get("extracted_text") or "").strip()
    vision_succeeded = vision_status == "ok" or bool(vision_description or vision_text)

    meaningful: list[str] = []
    decorative: list[str] = []
    ledger: list[dict[str, Any]] = []

    def note(bucket: list[str], message: str, signal: str) -> None:
        bucket.append(message)
        ledger.append(
            {
                "signal": signal,
                "weight": EVIDENCE_WEIGHTS.get(signal, 0.5),
                "supports": "meaningful" if bucket is meaningful else "decorative",
                "detail": message,
            }
        )

    # --- strong, content-bearing evidence -----------------------------------
    if text_signal["has_percent"]:
        note(meaningful, "contains percentage values", "percentages")
    if text_signal["has_currency"]:
        note(meaningful, "contains currency values", "currency")
    if text_signal["has_units"]:
        note(meaningful, "contains numeric units", "units")
    if text_signal["distinct_numeric_count"] >= _STRONG_NUMBER_COUNT:
        note(meaningful, "contains multiple numeric values", "numeric_values")
    if text_signal["chart_terms"] or text_signal["chart_phrases"]:
        note(meaningful, "contains chart-related vocabulary", "chart_vocabulary")
    if text_signal["diagram_terms"]:
        note(meaningful, "contains diagram-related vocabulary", "diagram_vocabulary")
    if text_signal["table_like"]:
        note(meaningful, "contains a table-like structure", "table_structure")
    if text_signal["domain_terms"]:
        note(meaningful, "contains domain vocabulary", "domain_vocabulary")
    if text_signal["content_word_count"] >= 1:
        note(meaningful, "contains meaningful text labels", "text_labels")
    if text_signal["characters"] >= _SUBSTANTIAL_OCR_CHARACTERS or text_signal["words"] >= _SUBSTANTIAL_OCR_WORDS:
        note(meaningful, "contains substantial OCR text", "substantial_text")
    if has_chart_data:
        note(meaningful, "carries chart metadata", "chart_data")
    if _chart_annotation(record):
        note(meaningful, "carries chart metadata", "chart_annotation")
    if docling_class and any(
        term in docling_class for term in ("chart", "graph", "diagram", "plot", "infographic")
    ):
        note(meaningful, f"classified as {docling_class} by Docling", "docling_chart_class")
    if docling_class and any(term in docling_class for term in ("molecule", "formula")):
        note(meaningful, "carries scientific structure metadata", "chart_data")
    if vision_succeeded and vision_category in VISION_MEANINGFUL_CATEGORIES:
        note(meaningful, "Vision classified it as document content", "vision_classification")
    elif vision_succeeded and vision_description and text_signal["empty"]:
        note(meaningful, "Vision returned a description", "vision_description")

    # --- strong, decorative evidence ----------------------------------------
    small_enough = coverage <= _BRAND_IMAGE_MAX_COVERAGE and visual.get("area", 0) <= _BRAND_IMAGE_MAX_AREA
    logo_shaped = bool(
        visual.get("thin_band") and (visual.get("height") or 0) <= 64
    ) or bool(visual.get("tiny"))
    if text_signal["brand_only"] and small_enough:
        note(decorative, "text content is branding only", "branding_only")
    if text_signal["brand_only"] and band in repeated_bands and small_enough:
        note(decorative, "branding repeated in the same page region", "branding_repeated")
    if text_signal["page_number_only"] and small_enough:
        note(decorative, "text content is a page number", "page_number")
    digest = record.get("sha256")
    if digest and str(digest) in repeated_assets:
        note(decorative, "identical asset repeated on other pages", "repeated_asset")
    if text_signal["empty"] and not has_chart_data and not vision_succeeded:
        if band in repeated_bands and small_enough and (logo_shaped or visual.get("tiny")):
            note(decorative, "no text and repeated page furniture in the same region", "empty_asset")
        elif visual.get("flat") or visual.get("near_blank"):
            note(decorative, "no text and visually empty asset", "empty_asset")
        elif visual.get("tiny"):
            note(decorative, "no text and icon-sized asset", "icon_sized")
    if background and text_signal["empty"] and not has_chart_data:
        note(decorative, "page-sized background without independent semantics", "page_background")

    # --- full page image that only duplicates the page text layer (CHANGE 4) --
    duplicate = dict(full_page_duplicate or {})
    if duplicate.get("duplicate") and not has_chart_data:
        note(
            decorative,
            "page-sized image repeats the page text layer "
            f"({int(100 * float(duplicate.get('token_coverage') or 0.0))}% of its tokens already present)",
            "text_layer_duplicate",
        )

    if not saved and not meaningful:
        note(decorative, "image asset could not be extracted", "missing_asset")

    meaningful = sorted(set(meaningful))
    decorative = sorted(set(decorative))
    # Conservative default: an image is only dropped from Markdown when there is
    # positive evidence that it is branding/decoration. Absence of evidence is
    # not evidence of absence of information.
    is_meaningful = bool(meaningful) or not decorative
    if duplicate.get("duplicate") and not has_chart_data:
        # A page-sized image whose text is already in the page text layer is
        # redundant by construction: every other signal above is describing
        # content that the Markdown already carries as text. Only structured
        # chart data (values the text layer does not contain) can rescue it.
        is_meaningful = False
    if is_meaningful and not meaningful:
        meaningful = ["no decorative evidence found; preserved by default"]
    semantic_class, class_reason = _semantic_class(
        record=record,
        text_signal=text_signal,
        docling_class=docling_class,
        classification_terms=classification_terms,
        has_chart_data=has_chart_data,
        vision_category=vision_category if vision_succeeded else "",
        background=background,
        meaningful=is_meaningful,
        decorative=decorative,
    )
    method = _classification_method(
        class_reason=class_reason,
        docling_class=docling_class,
        vision_succeeded=vision_succeeded,
        has_meaningful_evidence=bool(meaningful) and bool(meaningful != ["no decorative evidence found; preserved by default"]),
    )
    confidence, band, method = score_confidence(ledger, method=method)
    return {
        "meaningful": is_meaningful,
        "semantic_class": semantic_class,
        "classification_reason": class_reason,
        "classification_confidence": confidence,
        "classification_confidence_band": band,
        "classification_method": method,
        "evidence_ledger": ledger,
        "evidence_weight": round(
            sum(float(item.get("weight") or 0.0) for item in ledger), 3
        ),
        "meaningful_evidence": meaningful,
        "decorative_evidence": decorative,
        "signals": {
            "text": text_signal,
            "visual": visual,
            "position": position,
            "context": {
                "band_repeated": band in repeated_bands if band else False,
                "branding_vocabulary": vocabulary,
                "repeated_asset": bool(digest and str(digest) in repeated_assets),
            },
        },
    }


def _classification_method(
    class_reason: str,
    docling_class: str | None,
    vision_succeeded: bool,
    has_meaningful_evidence: bool,
) -> str:
    """Explain which layer decided the semantic class."""
    if class_reason.startswith("vision_"):
        return METHOD_VISION
    if class_reason.startswith("docling_"):
        return METHOD_DOCLING
    if class_reason == "no_signal":
        return METHOD_DEFAULT
    if vision_succeeded and not has_meaningful_evidence:
        return METHOD_VISION
    return METHOD_DETERMINISTIC


def _semantic_class(
    record: Mapping[str, Any],
    text_signal: Mapping[str, Any],
    docling_class: str | None,
    classification_terms: str,
    has_chart_data: bool,
    vision_category: str,
    background: bool,
    meaningful: bool,
    decorative: Sequence[str] = (),
) -> tuple[str, str]:
    """Return ``(semantic_class, reason)``.

    ``reason`` is a short machine-readable key naming the signal that decided
    the class; it is what makes a classification explainable and what the
    confidence score is anchored on.
    """
    if not meaningful:
        if any("page-sized image repeats the page text layer" in item for item in decorative):
            return SEMANTIC_DUPLICATE, "text_layer_duplicate"
        if text_signal["brand_only"] or any("branding" in item for item in decorative):
            return SEMANTIC_LOGO, "branding_signal"
        if text_signal["page_number_only"]:
            return SEMANTIC_PAGE_NUMBER, "page_number_signal"
        if background:
            return SEMANTIC_BACKGROUND, "background_signal"
        return SEMANTIC_DECORATION, "decoration_signal"
    chart_hint = any(
        term in classification_terms
        for term in ("chart", "graph", "plot", "pie", "bar", "line", "scatter", "infographic")
    )
    diagram_hint = any(
        term in classification_terms
        for term in ("diagram", "flow", "schematic", "network", "matrix", "process")
    )
    numeric_chart = (
        text_signal["has_percent"]
        or text_signal["distinct_numeric_count"] >= _STRONG_NUMBER_COUNT
        or text_signal["chart_phrases"]
    )
    if has_chart_data or _chart_annotation(record) or chart_hint:
        if has_chart_data or _chart_annotation(record):
            return SEMANTIC_CHART, "structured_chart_metadata"
        return SEMANTIC_CHART, "chart_vocabulary"
    if text_signal["has_percent"] and text_signal["distinct_numeric_count"] >= _STRONG_NUMBER_COUNT:
        return SEMANTIC_CHART, "numeric_chart_signal"
    if float(record.get("coverage") or 0.0) >= _PAGE_SCALE_COVERAGE and text_signal["words"] >= 3:
        return SEMANTIC_TEXT, "page_scale_text"
    if diagram_hint or (numeric_chart and text_signal["chart_terms"]):
        return SEMANTIC_CHART, "chart_vocabulary"
    if text_signal["diagram_terms"]:
        return SEMANTIC_DIAGRAM, "diagram_vocabulary"
    if text_signal["table_like"] and not numeric_chart:
        return SEMANTIC_TABLE, "table_structure"
    if docling_class == "other":
        return SEMANTIC_CHART, "docling_class_other"
    if vision_category == "diagram_chart":
        return (SEMANTIC_CHART, "vision_diagram_chart") if numeric_chart else (
            SEMANTIC_DIAGRAM,
            "vision_diagram_chart",
        )
    if vision_category == "text_image":
        return SEMANTIC_TEXT, "vision_text_image"
    if vision_category == "photo":
        return SEMANTIC_PHOTO, "vision_photo"
    if text_signal["table_like"]:
        return SEMANTIC_TABLE, "table_structure"
    if text_signal["words"]:
        return (
            (SEMANTIC_TEXT, "text_only_labels")
            if text_signal["numeric_count"] == 0
            else (SEMANTIC_CHART, "numeric_labels")
        )
    return SEMANTIC_UNCLASSIFIED, "no_signal"


def markdown_noun(semantic_class: str | None) -> str:
    """Return a human-readable noun for a semantic class."""
    return {
        SEMANTIC_CHART: "chart",
        SEMANTIC_DIAGRAM: "diagram",
        SEMANTIC_TABLE: "table image",
        SEMANTIC_TEXT: "text image",
        SEMANTIC_PHOTO: "image",
        SEMANTIC_LOGO: "logo",
        SEMANTIC_PAGE_NUMBER: "page number",
        SEMANTIC_DECORATION: "decorative graphic",
        SEMANTIC_BACKGROUND: "page background",
        SEMANTIC_DUPLICATE: "duplicate",
    }.get(str(semantic_class or ""), "image")
