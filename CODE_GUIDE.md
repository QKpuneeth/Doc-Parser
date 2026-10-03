# Code Guide — Docling Slim Parser

A reading guide for `parse_docling_slim.py` and friends. It exists to answer three
questions: *what does this program produce*, *how does the data flow through it*, and
*where do I change a given behaviour*.

- [1. What this is](#1-what-this-is)
- [2. Quick start](#2-quick-start)
- [3. The output contract](#3-the-output-contract)
- [4. Module map](#4-module-map)
- [5. The one data structure that matters: the image record](#5-the-one-data-structure-that-matters-the-image-record)
- [6. Reading order](#6-reading-order)
- [7. Flow diagram](#7-flow-diagram)
- [8. The image decision pipeline in detail](#8-the-image-decision-pipeline-in-detail)
- [9. Concepts you must not confuse](#9-concepts-you-must-not-confuse)
- [10. Where to change what](#10-where-to-change-what)
- [11. Tests](#11-tests)
- [12. Invariants and gotchas](#12-invariants-and-gotchas)

---

## 1. What this is

A single-process document parser that turns a **PDF, DOCX, or PPTX** into a
**RAG-ready Markdown document** plus auditable sidecar metadata.

It is a *slim* wrapper around Docling: Docling does the layout analysis, OCR, and
Markdown export, and this code adds the things Docling does not do — image
classification with a confidence score, an explicit keep/omit decision for every
image, optional Vision-LLM enrichment, and a self-verifying output bundle.

It deliberately does **not** chunk, embed, or write to a vector store. It produces
files and returns a dictionary; that is the whole product.

```
input.pdf ──► Docling ──► blocks (text / picture / table)
                            │
                            ├─► images/     classified, scored, kept or omitted
                            ├─► document.md  the deliverable
                            └─► metadata.json / canonical.json / manifest.json
```

---

## 2. Quick start

```bash
# CLI
venv/bin/python parse_docling_slim.py input.pdf --output out_dir --force
venv/bin/python parse_docling_slim.py input.docx --output out --emit-canonical --force
venv/bin/python parse_docling_slim.py --help          # every flag, with defaults

# Batch: several files, a directory, or a zip → one subdirectory per document
venv/bin/python parse_docling_slim.py study_a.pdf study_b.pdf --output out --force
venv/bin/python parse_docling_slim.py studies/              --output out --force
venv/bin/python parse_docling_slim.py studies.zip           --output out --force

# Python API — returns the full canonical dict in memory
from parse_docling_slim import parse_document, parse_many
result = parse_document("input.pdf", output_dir="out", emit_canonical=False)
result["stats"]["images_in_markdown"]     # 50
result["validation"]["status"]            # "pass"

# parse_many resolves files/directories/zips into documents; a single document
# lands flat in output_dir (identical to parse_document), several get subdirs.
summary = parse_many(["studies.zip"], output_dir="out", force=True)
summary["succeeded"] / summary["failed"]  # one row per document

# Tests (unittest, not pytest)
venv/bin/python -m unittest tests.test_parse_docling_slim          # 120 tests, ~35 s
venv/bin/python -m unittest tests.test_parse_docling_slim.VisionConfidenceGateTests
venv/bin/python -m unittest tests.test_parse_docling_slim._BatchInputTests
```

`parse_document()` returns the complete canonical dict **regardless of what is
written to disk**. `emit_canonical=False` controls only whether the file is
persisted. This is deliberate: the CLI is disk-oriented, library callers want the
object graph.

---

## 3. The output contract

Everything downstream reads these files. If you change one, you are changing a
public interface.

```
out_dir/
├── document.md              # THE deliverable: RAG-ready Markdown
├── metadata.json            # compact retrieval index  (default ON)
├── canonical.json           # full parser internals   (opt-in, off by default)
├── manifest.json            # what was written + checksums + validation verdict
├── images/
│   ├── image_0001.png …     # every extracted asset, KEPT OR NOT
│   └── table_0001.png …     # table renderings
└── logs/
    ├── parser.log
    └── image_decisions.jsonl # one JSON object per image: the audit trail
```

| File | Answers | Written when |
|---|---|---|
| `document.md` | "What should the model read?" | always |
| `metadata.json` | "Which images exist, and should I fetch them?" | `emit_metadata=True` (default) |
| `canonical.json` | "Why did the parser do that?" | `emit_canonical=True` (opt-in) |
| `manifest.json` | "Did the run actually produce what it claims?" | always |
| `logs/image_decisions.jsonl` | "Why was *this* image dropped?" | any image exists |

Three rules make the bundle trustworthy, and all three are enforced by code:

1. **Omitted images are still written to disk.** Omission is a *Markdown* decision,
   not a deletion. A multimodal consumer that wants the logo can still fetch it.
2. **Every path in the manifest exists, and every file on disk is in the manifest.**
   Verified with SHA-256 after the run (`_verify_manifest_contents`, line 2704).
3. **`document.md` contains no base64 and no absolute paths.** Image refs are
   always relative (`images/image_0001.png`).

`metadata.json` is the file a retrieval system should actually use. Per image it
carries `path`, `page`, `type`, `confidence`, `confidence_band`,
`included_in_markdown`, `omission_reason`, `page_coverage`, `asset_sha256`,
`occurrences`, `caption`, `ocr_text`, `vision`, and an `evidence` ledger of the
facts behind the classification.

---

## 4. Module map

| File | Lines | Responsibility |
|---|---|---:|---|
| `parse_docling_slim.py` | 3155 | Everything: config, extraction, classification wiring, Vision, Markdown, metadata, validation, manifest, CLI, batch/zip fan-out (`parse_many`) |
| `image_signals.py` | 1055 | Pure signal extraction + scoring. No I/O, no Docling import, no side effects |
| `vision_utils.py` | 216 | Optional OpenAI/Azure call. Returns `None` cleanly when unconfigured |
| `tests/test_parse_docling_slim.py` | 2065 | 120 tests, 17 classes |

The split between the first two files is the important architectural boundary:

> **`image_signals.py` decides what it *sees*. `parse_docling_slim.py` decides what
> it *does about it*.**

`image_signals.py` is pure and trivially testable — every function takes a record or
a PIL image and returns a dict. All I/O, all policy, all thresholds that depend on
`ParserConfig` live in the parser. If you find yourself adding a `if config...` to
`image_signals.py`, it probably belongs in the parser.

`vision_utils.py` is deliberately inert by default: no API key means
`vision_available()` is `False` and the parser skips enrichment with a recorded
reason instead of failing.

---

## 5. The one data structure that matters: the image record

Almost every behaviour in this codebase is a function of the **image record** — a
plain `dict` created in the extraction loop (line 2252) and progressively enriched
until it is rendered, validated, and exported.

```python
{
  # identity & provenance
  "ref": "#/pictures/3",          # Docling's own reference
  "page": 13, "order": 42, "level": 2,
  "source": {"page": 13, "bbox": [...], ...},   # from Docling provenance
  "page_inferred": False,         # True if the page was guessed, not reported

  # asset
  "path": "images/image_0016.png", "relative_path": ..., "saved": True,
  "width": 1676, "height": 953, "asset_sha256": "…",
  "duplicate_of": None, "occurrences": 1,

  # measured signals
  "visual": {...},                # pixel facts: colours, edges, aspect, tiny/blank
  "position": {...},              # band: top-left, middle-center, …
  "coverage": 0.81,               # fraction of page area
  "background": True,
  "ocr": "Phase 1\nPhase 2a\n…",  # text nested inside the picture
  "ocr_characters": 1581,

  # decisions
  "semantic_class": "chart",
  "classification_method": "deterministic",
  "classification_reason": "…",
  "confidence": 0.93, "confidence_band": "high",
  "include_in_md": True,
  "omit_reason": None,            # set only when excluded
  "vision": {...}, "vision_status": "disabled", "vision_error": None,
}
```

Two conventions to internalise:

- **`include_in_md` is the single source of truth.** Every other function reads it
  and never recomputes it. Rendering, validation, statistics, and metadata all
  derive from this one boolean.
- **`omit_reason` is absent, not `None`, when the image is kept.** `canonical.json`
  consumers can use key presence as the signal. The metadata sidecar normalises this
  to `null` for JSON friendliness.

The record is created once and then passed, by reference, through a fixed sequence of
enrichment stages. No stage reads another's output from the filesystem; they all
mutate and re-read the same list.

---

## 6. Reading order

Read it in this order. Each step only makes sense given the previous one, and the
line numbers are into `parse_docling_slim.py` unless stated otherwise.

**1 — The contract (10 min)**
`ParserConfig` (58) → all thresholds and switches live here. `parse_document` (2080)
→ the public API and its argument validation. This is the surface you must stay
compatible with.

**2 — The pipeline (30 min)**
`_parse_into` (2212) is the spine: ~490 lines that read top-to-bottom as the actual
data flow. Read it once end to end with §7 open beside it. This single function is
worth more than any other part of the file.

**3 — The decisions that matter (60 min)**
- `_classify_images` (1014) — the orchestrator for image policy
- `image_signals.py::classify_image` (741) — how a class is chosen
- `image_signals.py::score_confidence` (584) — how the 0–1 score is produced
- `_full_page_duplicate_signals` (1073) — the page-duplicate rule and its guard
- `_apply_vision` (1211) — confidence gating, retries, budget accounting

**4 — The output (30 min)**
`_image_block` (675) → what an image looks like in Markdown. `_render_document_markdown`
(812) → page assembly. `_build_metadata` (1460) → the sidecar. `_validate_outputs`
(1571) → the self-check.

**5 — The safety net (as needed)**
`_prepare_output` / `_make_working_dir` / `_finalize_output` (192–237) implement
atomic output: everything is built in a temp directory and swapped into place only
after the parse succeeds, so a crash never leaves a half-written bundle.

Skip lines 239–1070 on first read. They are extraction helpers — one function per
small question ("what is this item's page?", "how big is this picture?"). They are
easy to read on demand.

---

## 7. Flow diagram

### End to end

```
                    ┌─────────────────────────────────────────┐
   parse_document   │ validate args → ParserConfig → output dir │
   (2080)           └────────────────────┬────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │  _make_working_dir     │  build in a temp dir
                            │  (atomic output)       │  so a crash cannot
                            └───────────┬────────────┘  corrupt real output
                                        │
                                        ▼
   ╔══════════════════════════════════════════════════════════════════════╗
   ║                        _parse_into  (2212)                          ║
   ║                                                                      ║
   ║  ① build_converter(93) → Docling parses the file                     ║
   ║  ② _item_lookup(379)      index every item by self_ref               ║
   ║  ③ _refs_nested_in_pictures(423)  text that lives INSIDE a picture   ║
   ║                                                                      ║
   ║  ④ ONE walk over every item (2252) → records                         ║
   ║        PictureItem → save PNG, gather visual/position/OCR signals     ║
   ║        TableItem   → rows + markdown + optional rendering             ║
   ║        text        → text block  (page_text_by_page, *non-nested*)   ║
   ║                                                                      ║
   ║  ⑤ _classify_images(1014)      class + score + keep/omit             ║
   ║  ⑥ _link_duplicate_assets(539) one decision per asset occurrence     ║
   ║  ⑦ _apply_vision(1211)        optional LLM enrichment, gated         ║
   ║  ⑧ _write_image_decision_log(123)  audit trail (JSONL)               ║
   ║  ⑨ _render_document_markdown(812)  document.md                       ║
   ║                                                                      ║
   ║  ⑩ write metadata.json / canonical.json (per config)                 ║
   ║  ⑪ _validate_outputs(1571)    check the files just written           ║
   ║  ⑫ manifest.json + _verify_manifest_contents(2704)  checksums        ║
   ║                                                                      ║
   ║  → raise if validation FAILED, else return the canonical dict        ║
   ╚══════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │  _finalize_output      │  atomic swap
                            │  _refresh_manifest_    │  checksums after
                            │      checksums(2172)   │  logging flushes
                            └────────────────────────┘
```

Stages ⑩–⑫ are ordered deliberately: the files are written **before** validation, so
validation inspects what actually landed on disk rather than what was intended, and
the verdict is then folded back into the sidecars.

### One image, end to end

```
   Docling PictureItem
          │
          ▼
   ┌──────────────────┐
   │ save the PNG     │──► images/image_0016.png   (written even if later omitted)
   │ measure geometry │    width/height/coverage
   └────────┬─────────┘
            ▼
   ┌──────────────────────────────────────────────────┐
   │ _classify_images                                  │
   │                                                  │
   │  visual_signals    → colours, edges, tiny, blank │
   │  text_signals      → numbers, units, brand words │
   │  position_signals  → band, size, repeat counts   │
   │          │                                       │
   │          ▼                                       │
   │  classify_image()   → "chart"                    │
   │  score_confidence() → 0.93 (high)                │
   │                                                  │
   │  page-sized? compare OCR to the page text layer  │
   │      └─ identical AND the text is really exported │
   │         in the Markdown?  no  →  keep the image   │
   │                                                  │
   │  decorative class?  → include_in_md = False      │
   │                     omit_reason = "decorative_or_logo"
   └────────┬─────────────────────────────────────────┘
            ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ kept             │   │ omitted          │   │ both             │
   │ ![alt](path)     │   │ <!-- image       │   │ → metadata.json  │
   │ **Image type:**  │   │      omitted: …  │   │ → JSONL audit    │
   │ **OCR text:**    │   │      -->         │   │ → statistics     │
   │ **Visual desc:** │   │ (chunker-invisible)  │ │
   └──────────────────┘   └──────────────────┘   └──────────────────┘
```

Note the shape of the omitted branch: an **HTML comment**, not a visible note. A
downstream chunker splitting on Markdown must not mistake it for content, and a
human reading the document should not see a wall of "I dropped this" notices. The
machine-readable version of the same decision lives in `metadata.json`.

---

## 8. The image decision pipeline in detail

This is where the interesting logic and all the risk live. Understand §5 (the record)
first; every stage below mutates it.

### 8.1 Classification — "what kind of thing is this?"

`image_signals.classify_image` (741) picks one of ten semantic classes:
`chart`, `diagram`, `table_image`, `text_image`, `photo`, `logo`, `page_number`,
`icon_or_decoration`, `background`, `unclassified_image` (plus `duplicate`, added
later by the parser).

It blends three independent evidence sources, in descending authority:

1. **Docling's own label** — a pre-trained layout model. Usually right.
2. **Structured chart data** — if the picture carries real numeric cells, it is a
   chart regardless of what any OCR says.
3. **Hand-rolled signals** — `text_signals` (percentages, currency, units, brand
   vocabulary), `visual_signals` (flat colours ⇒ chart, few colours + tiny ⇒ logo,
   sparse ink ⇒ decoration), `position_signals` (band, repeat counts, the document
   context from `build_document_context`).

The result is written as `classification_method` so a consumer can tell a Docling
label from a heuristic guess.

### 8.2 Confidence — "how sure are we?"

`score_confidence` (584) produces 0–1 plus a band (`high` ≥ 0.8, `medium` ≥ 0.5,
`low` below). Every decision that could drop content is gated on this number. It
exists so that a downstream system can set its own risk tolerance without
reimplementing the heuristics.

`evidence` (and the backwards-compatible `meaningful_evidence`) lists the concrete
facts that produced the score — e.g. `["ocr_contains_numbers", "flat_colour_palette"]`.
This is the difference between "0.87" and "0.87 because it is a flat-colour graphic
whose only words are a brand name".

### 8.3 Omission — "does the reader need it?"

Three rules can set `include_in_md = False`, and they are checked in this order:

| Rule | `omit_reason` | Why it is safe |
|---|---|---|
| Decorative class | `decorative_or_logo` | Logos, icons, page numbers, rules — no information a reader needs |
| Page-sized text duplicate | `full_page_image_duplicates_text_layer` | The words are already in the page's text layer |
| Vision-disabled low confidence | *(kept, flagged)* | Never omits; only enriches |

The page-duplicate rule is the subtle one, and it carries the single most important
guard in the codebase:

> **The Markdown is exported with `traverse_pictures=False`, so text nested inside a
> picture is never emitted on its own.** Our image block is the only thing that puts
> those words in the document.

So when comparing an image's OCR against "the page text layer", only text that is
genuinely *not* nested in a picture may count. `_refs_nested_in_pictures` (423)
computes that set, and nested text is excluded from the comparison at line 2250.

Getting this wrong is silent and severe. On the 45-page acceptance document, 28
page-sized figures hold 105 text items each, all nested inside the picture. Treating
that as a text layer let the rule omit all 28 — and `document.md` silently lost 2010
text blocks, with page 13 reduced to a 94-byte comment and nothing else. The tests
passed the whole time. Only a whole-document text-parity check caught it.

### 8.4 Deduplication — "is this the same asset twice?"

`_link_duplicate_assets` (539) groups records by SHA-256 of the decoded pixels, then
gives every occurrence of an asset the owner's decision, and reports
`duplicate_of` / `occurrences` / `pages`.

This is **byte-exact, not perceptual**. On the acceptance document it never fires:
all 64 images hash differently, because Docling re-encodes an embedded image
slightly differently for each occurrence. Perceptual hashing would catch those, at
the cost of occasionally merging genuinely different charts. Exact hashing was kept
because it is predictable and never destroys a distinct image.

### 8.5 Vision — "can an LLM add anything?"

Off by default. When enabled, `_apply_vision` (1211):

- only enriches images whose confidence is **below** `vision_confidence_threshold`
  (default 0.75) — confident classifications are not worth an API call;
- respects `vision_max_images` and reports `limit_reached` when it binds;
- retries only *transient* failures (timeouts, 429, 5xx) and records
  `vision_attempts`;
- never fails the run. A Vision error downgrades that image's status and nothing else.

Cost and token usage land in the statistics, so a run that starts spending money is
visible after the fact.

### 8.6 Validation and the manifest

`_validate_outputs` (1571) checks the produced files: the Markdown exists, every
referenced asset exists, no absolute paths, confidence bands are valid, duplicate
links are consistent, metadata agrees with the Markdown, and the canonical file is
present iff it was requested. Its verdict (`pass` / `fail` / `skipped`) is folded
into `metadata.json` and `canonical.json`.

`manifest.json` is then built from **what was actually written** — not a fixed list —
and `_verify_manifest_contents` (2704) checksums every artifact and reports anything
on disk that the manifest does not list. Checksums are recomputed one last time by
`_refresh_manifest_checksums` (2172) after logging flushes, since the log file is
itself an artifact.

---

## 9. Concepts you must not confuse

| Concept | Meaning | Not to be confused with |
|---|---|---|
| `semantic_class` | *What the image is* — chart, logo, photo… | `include_in_md` — whether it is in the Markdown |
| `confidence` | *How sure the classification is* | `coverage` — how much of the page it fills |
| `coverage` | Image area ÷ page area | `page_coverage` in metadata, same number, different namespace |
| `background` | Covers ≥ 60% of the page | Being decorative; a background chart is real content |
| `omitted` | Not referenced in `document.md` | *Deleted*; the file is still on disk |
| `native OCR` | Text Docling read from the picture | Vision output; never labelled as Vision |
| `Vision` | LLM description of the image | OCR; never labelled as OCR |
| `text layer` | Page text actually exported to Markdown | Text nested inside a picture, which is **not** exported |
| `metadata.json` | Compact retrieval index | `canonical.json` — full internals incl. `config` and `blocks` |

The last row of that table is the bug that cost the most time in this project. Keep
it in mind whenever "the text is already in the document" appears in a comment.

---

## 10. Where to change what

| To change… | Edit | Around line |
|---|---|---:|
| Any threshold or default | `ParserConfig` | 58 |
| A CLI flag | `_build_cli` | 2739 |
| Image keep/omit policy | `_classify_images` | 1014 |
| A semantic class or its evidence | `image_signals.classify_image` | 741 |
| The confidence score | `image_signals.score_confidence` | 584 |
| What an image looks like in Markdown | `_image_block` | 675 |
| The page-duplicate rule / its guard | `_full_page_duplicate_signals`, `_refs_nested_in_pictures` | 1073, 423 |
| Duplicate-asset linking | `_link_duplicate_assets` | 539 |
| Vision gating, retries, budget | `_apply_vision` | 1211 |
| The Vision API call itself | `vision_utils.classify_and_extract_image` | 139 |
| `metadata.json` contents | `_build_metadata`, `_image_metadata` | 1460, 1387 |
| Validation checks | `_validate_outputs` | 1571 |
| Manifest contents | the `manifest = {…}` block | 2626 |
| Pipeline order | `_parse_into` | 2212 |
| Atomic output behaviour | `_prepare_output` … `_finalize_output` | 192–237 |

Schema version: `SCHEMA_VERSION` (line ~40) is used by `metadata.json`,
`canonical.json`, and `manifest.json` together. Bump it **only** for a breaking shape
change; today it is `"4.0"`.

Feature work is tagged in the source with `CHANGE n:` comments that name the intent
behind non-obvious guards. Keep that convention — several of those comments are the
only record of a bug that was already paid for once.

---

## 11. Tests

`venv/bin/python -m unittest tests.test_parse_docling_slim` → 91 tests, ~35 s.

| Class | Covers |
|---|---|
| `DecorativeDetectionTests` | logo / icon / page-number omission |
| `NestedPictureTextTests` | text-inside-a-picture is not a text layer (the 2010-block regression) |
| `FullPageDuplicateDetectionTests` | the page-duplicate rule and its thresholds |
| `MarkdownImagePolicyTests` | image block format, empty sections, omission notes |
| `VisionUtilsTests` | unconfigured-provider behaviour |
| `VisionDecisionTests` | when Vision runs and when it is skipped |
| `VisionConfidenceGateTests` | threshold gating, retries, budget, transient errors |
| `AssetDeduplicationTests` | occurrence linking and propagated decisions |
| `MetadataAndDecisionLogTests` | the real artefact set from a generated PDF |
| `ValidationTests` | each validation rule and its failure mode |
| `TablePolicyTests` | simple vs complex tables |
| `OfficeFixtureTests` | DOCX / PPTX end to end |
| `SourceFingerprintTests` | provenance and atomic output |
| `MarkdownReferenceTests` | refs resolve, paths stay relative |

Two conventions worth copying when you add tests:

- **Unit tests build a record by hand** — `make_record(**overrides)` at line 32 gives
  you a valid image record with sane defaults. Prefer this; it is fast and pins one
  decision at a time.
- **Integration tests parse a real generated PDF** — a 2-page fixture built with
  PyMuPDF, so the assertions cover the actual artefact bundle rather than mocks.

One trap, learned the hard way: a fake must model the *real* shape of the thing you
depend on. An early version of `NestedPictureTextTests` used nodes where Docling
actually stores `RefItem` references, so the guard it tested matched nothing on a
real document while the test stayed green. When a test encodes a fact about Docling's
internals, verify that fact against a converted document before trusting the test.

---

## 12. Invariants and gotchas

Things that will bite you, in rough order of how expensive they were to learn.

1. **A picture is the only carrier of its nested text.** `export_to_markdown` is
   called with `traverse_pictures=False`, so text inside a picture never reaches the
   document on its own. Never treat nested text as an already-present text layer.
2. **Whole-document text parity is the only safety net.** Per-page and per-image
   checks all passed while 2010 text blocks silently vanished. When changing anything
   that can omit content, diff the native text blocks of a full parse against the
   Markdown — the acceptance baseline is in `/tmp/opencode/pre_change_output`.
3. **`_render_markdown` asserts marker count == record count** (line 715). If you
   add an image record, the placeholder in the exported Markdown must line up with it
   or the run raises. This is a feature: it is the invariant that keeps images and
   their text in the right order.
4. **Omission never deletes the asset.** Do not "optimise" by skipping the write for
   omitted images; the retrieval side depends on being able to fetch them.
5. **Checksums are computed after logging finishes.** The log file is an artifact, so
   it changes size while being written. `_refresh_manifest_checksums` exists for this.
6. **Output is atomic.** Everything is built in `_make_working_dir` and swapped by
   `_finalize_output`. Never write directly into the final directory.
7. **`parse_document` returns the canonical dict even when the file is not written.**
   Changing that would break library callers.
8. **Docling's `iterate_items(traverse_pictures=True)` yields nested text, but the
   Markdown export does not.** That asymmetry is the root of gotcha 1.
9. **Memory.** The 45-page acceptance document peaks near 3.3 GB RSS. Convert one
   document at a time; if the host is tight, use the guarded runner rather than the
   raw CLI.

---

## Appendix — verified behaviour on the acceptance document

`TEST.pdf`, 45 pages, 2729 text blocks, 54,976 text characters, 3 tables, 64 images.

| | Value |
|---|---|
| Validation | `pass` |
| Manifest | `complete`, 71 artifacts, 0 integrity errors, 0 unlisted files |
| Images in `document.md` | 50 |
| Images omitted | 14, all `decorative_or_logo` |
| Confidence bands | 44 high, 13 medium, 7 low |
| Page-duplicate omissions | 0 — correctly inert, see §8.3 |
| Native text blocks lost | 0 (51 table-rendered blocks, same as the pre-change baseline) |
| `canonical.json` | absent by default; verified present with `--emit-canonical` |
