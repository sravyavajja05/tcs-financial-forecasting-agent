import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pymupdf

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
PERIOD_PATTERN = re.compile(r"\bQ([1-4])\s*FY\s*(\d{2,4})\b", re.IGNORECASE)


@dataclass(frozen=True)
class SourceDocument:
    text: str
    source_file: str
    source_url: str | None
    doc_type: str
    period: str | None
    page: int | None
    basis: str | None
    currency: str | None
    unit: str | None
    publication_date: str | None = None
    retrieval_status: str | None = None
    tables: tuple[tuple[tuple[str | None, ...], ...], ...] = ()


def _period_label(value: str | None) -> str | None:
    if not value:
        return None
    match = PERIOD_PATTERN.search(value)
    if not match:
        return None
    year = match.group(2)
    if len(year) == 4:
        year = year[-2:]
    return f"Q{match.group(1)} FY{year}"


class SourceRepository:
    """Loads local source documents and optional provenance metadata."""

    def __init__(self, data_dir: Path):
        self.root = data_dir.resolve()

    def load(self, doc_types: set[str] | None = None) -> list[SourceDocument]:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "source_manifest.json"
        manifest = self._read_manifest(manifest_path)
        documents: list[SourceDocument] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            relative = path.relative_to(self.root).as_posix()
            metadata = manifest.get(relative, {})
            doc_type = metadata.get("doc_type") or self._infer_doc_type(path)
            if doc_types is not None and doc_type not in doc_types:
                continue
            try:
                pages = self._read(
                    path, extract_tables=doc_type == "financial_report"
                )
            except Exception:
                logger.exception("Could not read source document %s", relative)
                raise
            for page_number, text, tables in pages:
                if not text.strip():
                    continue
                documents.append(
                    SourceDocument(
                        text=text,
                        source_file=relative,
                        source_url=metadata.get("source_url"),
                        doc_type=doc_type,
                        period=(
                            metadata.get("period")
                            or _period_label(path.name)
                        ),
                        page=page_number,
                        basis=metadata.get("basis"),
                        currency=metadata.get("currency"),
                        unit=metadata.get("unit"),
                        publication_date=metadata.get("publication_date"),
                        retrieval_status=metadata.get("retrieval_status"),
                        tables=tables,
                    )
                )
        return documents

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("source_manifest.json must contain a JSON object.")
        for file_name, metadata in payload.items():
            if not isinstance(file_name, str) or not isinstance(metadata, dict):
                raise ValueError("Manifest keys must be filenames with metadata objects.")
            if metadata.get("source_url") and not str(metadata["source_url"]).startswith(
                ("https://", "http://")
            ):
                raise ValueError(f"Invalid source_url for {file_name}.")
        return payload

    @staticmethod
    def _read(
        path: Path, *, extract_tables: bool = False
    ) -> list[
        tuple[
            int | None,
            str,
            tuple[tuple[tuple[str | None, ...], ...], ...],
        ]
    ]:
        if path.suffix.lower() == ".pdf":
            pages = []
            with pdfplumber.open(path) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        with pymupdf.open(path) as document:
                            text = document[index - 1].get_text()
                    page_tables = (
                        page.extract_tables() or []
                        if extract_tables
                        and "statement of financial results" in text.lower()
                        else []
                    )
                    tables = tuple(
                        tuple(tuple(cell for cell in row) for row in table)
                        for table in page_tables
                    )
                    pages.append((index, text, tables))
            return pages
        return [(None, path.read_text(encoding="utf-8"), ())]

    @staticmethod
    def _infer_doc_type(path: Path) -> str:
        name = path.name.lower()
        if "transcript" in name:
            return "transcript"
        if any(word in name for word in ("result", "financial", "report")):
            return "financial_report"
        return "unknown"
