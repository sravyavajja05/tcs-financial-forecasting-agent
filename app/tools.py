import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore

from app.sources import SourceDocument, SourceRepository

METRIC_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "revenue": (
        re.compile(r"\brevenue(?:\s+from\s+operations)?\b", re.IGNORECASE),
        re.compile(r"\btotal\s+revenue\b", re.IGNORECASE),
    ),
    "net_profit": (
        re.compile(r"\bnet\s+(?:profit|income)\b", re.IGNORECASE),
        re.compile(r"\bprofit\s+for\s+the\s+(?:period|quarter|year)\b", re.IGNORECASE),
    ),
    "operating_margin": (
        re.compile(r"\boperating\s+margin\b", re.IGNORECASE),
        re.compile(r"\bEBIT\s+margin\b", re.IGNORECASE),
    ),
}
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?:\(\s*\d[\d,]*(?:\.\d+)?\s*\)|[-+]?\d[\d,]*(?:\.\d+)?%?)"
)
PERIOD_PATTERN = re.compile(r"\bQ([1-4])\s*FY\s*(\d{2,4})\b", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", re.IGNORECASE
)
RAG_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "before", "being", "between", "both", "but", "can", "could", "did",
    "does", "during", "each", "from", "further", "had", "has", "have",
    "having", "here", "into", "its", "itself", "management", "more",
    "most", "other", "ours", "out", "over", "quarter", "same", "should",
    "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "until",
    "very", "was", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "would", "your", "tcs",
}


def _periods(text: str) -> list[str]:
    result = []
    for match in PERIOD_PATTERN.finditer(text):
        year = match.group(2)[-2:]
        result.append(f"Q{match.group(1)} FY{year}")
    return result


def _sort_period(label: str) -> tuple[int, int]:
    match = PERIOD_PATTERN.search(label)
    if not match:
        return (0, 0)
    return (int(match.group(2)[-2:]), int(match.group(1)))


class FinancialDataExtractorTool:
    """Extract quarterly financial evidence from PDF tables and text."""

    def __init__(self, data_dir: Path):
        self.repository = SourceRepository(data_dir)

    def extract(self, query: str = "", quarters: int = 4) -> dict:
        documents = [
            item
            for item in self.repository.load(
                {"financial_report", "financial_summary"}
            )
            if item.doc_type in {"financial_report", "financial_summary"}
        ]
        reports = [item for item in documents if item.doc_type == "financial_report"]
        summaries = [item for item in documents if item.doc_type == "financial_summary"]
        metrics: list[dict] = []
        report_period_sources: dict[str, SourceDocument] = {}

        for page in reports:
            basis = self._basis(page)
            if basis == "standalone" or not self._financial_results_page(page):
                continue
            if page.period:
                report_period_sources.setdefault(page.period, page)
            page_metrics = self._extract_tables(page, basis)
            extracted_names = {
                item["metric"] for item in page_metrics if item["value"] is not None
            }
            page_metrics.extend(
                self._extract_text_rows(page, basis, excluded=extracted_names)
            )
            metrics.extend(page_metrics)

        for page in summaries:
            metrics.extend(self._extract_reported_margins(page))

        existing = {
            (item["metric"], item["period"])
            for item in metrics
            if item["period"] != "period_unresolved"
        }
        for period, page in report_period_sources.items():
            basis = self._basis(page)
            for metric in ("revenue", "net_profit", "operating_margin"):
                key = (metric, period)
                if key not in existing:
                    metrics.append(
                        self._record(
                            page,
                            period,
                            metric,
                            None,
                            basis,
                            (
                                f"{metric.replace('_', ' ').title()} unavailable: "
                                "no attributable value was extracted from the "
                                "verified source pages."
                            ),
                            review_required=True,
                        )
                    )

        metrics = self._deduplicate_metrics(metrics)
        limitations: list[str] = []
        if not reports:
            limitations.append(
                "No verified financial report documents were found. Add actual TCS "
                "quarterly reports and a source_manifest.json entry before forecasting."
            )
        if not summaries:
            limitations.append(
                "No verified financial summary was available for a directly reported "
                "operating margin; no margin has been inferred."
            )
        if any(metric["source_url"] is None for metric in metrics):
            limitations.append(
                "One or more local reports have no source URL in source_manifest.json; "
                "their provenance has not been independently recorded."
            )
        if any(
            item["metric"] == "operating_margin"
            and item["basis"]
            and "excludes exceptional items" in item["basis"]
            for item in metrics
        ):
            limitations.append(
                "The Q1 FY27 operating margin is explicitly adjusted to exclude "
                "exceptional items and is not treated as comparable to unadjusted "
                "quarterly margins."
            )
        periods_seen = sorted(
            {item["period"] for item in metrics if item["period"] != "period_unresolved"},
            key=_sort_period,
        )
        if len(periods_seen) > quarters:
            keep = set(periods_seen[-quarters:])
            metrics = [
                item for item in metrics
                if item["period"] == "period_unresolved" or item["period"] in keep
            ]
        if any(item["review_required"] for item in metrics):
            limitations.append(
                "Values with missing or ambiguous period/source evidence are retained "
                "as unavailable evidence and excluded from numeric estimates."
            )
        metrics.sort(
            key=lambda item: (
                _sort_period(item["period"]),
                item["metric"],
                item["basis"] or "",
            )
        )
        return {
            "metrics": metrics,
            "quarter_over_quarter": self._quarterly_changes(metrics),
            "limitations": list(dict.fromkeys(limitations)),
        }

    @staticmethod
    def _basis(page: SourceDocument) -> str | None:
        text = page.text.lower()
        if "standalone" in text:
            return "standalone"
        if "consolidated" in text:
            return "consolidated"
        return page.basis

    @staticmethod
    def _financial_results_page(page: SourceDocument) -> bool:
        if page.source_file.lower().endswith(".pdf"):
            return "statement of financial results" in page.text.lower()
        return True

    @classmethod
    def _extract_tables(cls, page: SourceDocument, basis: str | None) -> list[dict]:
        results = []
        for table in page.tables:
            columns = cls._quarter_columns(table, page.period)
            if not columns:
                continue
            for row in table[2:]:
                if not row or not row[0]:
                    continue
                labels = [line.strip() for line in row[0].splitlines()]
                for label_index, label in enumerate(labels):
                    metric = cls._matching_metric(label)
                    if metric not in {"revenue", "net_profit"}:
                        continue
                    for column_index, period in columns:
                        if column_index >= len(row) or not row[column_index]:
                            continue
                        cell_lines = [
                            value.strip() for value in row[column_index].splitlines()
                        ]
                        if label_index >= len(cell_lines):
                            continue
                        raw_value = cell_lines[label_index]
                        value = cls._number_or_none(raw_value)
                        results.append(
                            cls._record(
                                page,
                                period,
                                metric,
                                value,
                                basis,
                                f"{label}: {raw_value}",
                                review_required=value is None or period == "period_unresolved",
                            )
                        )
        return results

    @staticmethod
    def _quarter_columns(
        table: tuple[tuple[str | None, ...], ...], fallback_period: str | None
    ) -> list[tuple[int, str]]:
        if len(table) < 2:
            return []
        group_row, date_row = table[0], table[1]
        selected = []
        group = ""
        for index, header in enumerate(group_row):
            if header:
                lowered = header.lower()
                if "three months ended" in lowered:
                    group = "quarter"
                elif "six months ended" in lowered or "year ended" in lowered:
                    group = "other"
            if index == 0 or group != "quarter":
                continue
            date = date_row[index] if index < len(date_row) else None
            period = FinancialDataExtractorTool._date_period(date or "")
            if period:
                selected.append((index, period))
        if not selected and fallback_period:
            selected.append((1, fallback_period))
        return selected

    @classmethod
    def _extract_text_rows(
        cls, page: SourceDocument, basis: str | None, excluded: set[str]
    ) -> list[dict]:
        lines = page.text.splitlines()
        results = []
        for index, line in enumerate(lines):
            metric = cls._matching_metric(line)
            if metric not in {"revenue", "net_profit", "operating_margin"}:
                continue
            if metric in excluded:
                continue
            if metric == "operating_margin" and page.source_file.lower().endswith(
                ".pdf"
            ):
                continue
            tokens = NUMBER_PATTERN.findall(line)
            if not tokens:
                continue
            values = [cls._number_or_none(token) for token in tokens]
            periods = cls._statement_periods(lines, index)
            context = "\n".join(lines[:index])
            quarter_periods = _periods(context)
            if len(periods) < min(len(values), 3):
                periods = quarter_periods[-len(values):]
            if len(values) == 1 and page.period:
                candidates = [(page.period, values[0])]
            elif periods:
                candidates = [
                    (periods[position], values[position])
                    for position in range(min(len(periods), len(values), 3))
                ]
            else:
                candidates = [(page.period or "period_unresolved", None)]
            for period, value in candidates:
                results.append(
                    cls._record(
                        page,
                        period,
                        metric,
                        value,
                        basis,
                        " ".join(line.split())[:600],
                        unit="%" if metric == "operating_margin" else None,
                        currency=None if metric == "operating_margin" else None,
                        review_required=value is None or period == "period_unresolved",
                    )
                )
        return results

    @classmethod
    def _statement_periods(cls, lines: list[str], row_index: int) -> list[str]:
        for heading_index in range(max(0, row_index - 40), row_index):
            if "three months ended" not in lines[heading_index].lower():
                continue
            for date_index in range(heading_index + 1, min(row_index, heading_index + 4)):
                date_match = re.search(
                    r"((?:[A-Za-z]+\s+\d{1,2},?\s*){2,})", lines[date_index]
                )
                if not date_match:
                    continue
                date_parts = re.findall(
                    r"([A-Za-z]+\s+\d{1,2}),?", date_match.group(1)
                )
                for year_line in lines[date_index + 1 : min(row_index, date_index + 3)]:
                    years = re.findall(r"\b(?:19|20)\d{2}\b", year_line)
                    if len(years) < 2:
                        continue
                    result = []
                    for date, year in zip(date_parts, years):
                        period = cls._date_period(f"{date} {year}")
                        if period:
                            result.append(period)
                    return result[:3]
        return []

    @classmethod
    def _extract_reported_margins(cls, page: SourceDocument) -> list[dict]:
        results = []
        pattern = re.compile(
            r"\b(?:operating\s+margin|EBIT\s+margin)\b"
            r"[^0-9%]{0,30}(-?\d+(?:\.\d+)?)\s*%",
            re.IGNORECASE,
        )
        for line in page.text.splitlines():
            match = pattern.search(line)
            if not match or re.search(r"\bFY\s*\d{2,4}\b", line[:match.start()], re.I):
                continue
            basis = cls._basis(page)
            if "*" in line and basis:
                basis = f"{basis} (excludes exceptional items)"
            results.append(
                cls._record(
                    page,
                    page.period or "period_unresolved",
                    "operating_margin",
                    float(match.group(1)),
                    basis,
                    " ".join(line.split())[:600],
                    unit="%",
                    currency=None,
                    review_required=page.period is None,
                )
            )
        return results

    @classmethod
    def _record(
        cls,
        page: SourceDocument,
        period: str,
        metric: str,
        value: float | None,
        basis: str | None,
        excerpt: str,
        *,
        unit: str | None = None,
        currency: str | None = None,
        review_required: bool = False,
    ) -> dict:
        if unit is None:
            unit = "%" if metric == "operating_margin" else page.unit
        if currency is None and metric != "operating_margin":
            currency = page.currency or (
                "INR" if "inr" in page.text.lower() or "₹" in page.text else None
            )
        if unit is None and metric != "operating_margin":
            unit = "crore" if re.search(r"\bcrore\b", page.text, re.I) else None
        return {
            "period": period,
            "metric": metric,
            "value": value,
            "unit": unit,
            "currency": currency,
            "basis": basis,
            "source_file": page.source_file,
            "source_url": page.source_url,
            "page": page.page,
            "excerpt": excerpt,
            "review_required": review_required,
        }

    @staticmethod
    def _date_period(value: str) -> str | None:
        match = DATE_PATTERN.search(value.replace("\n", " "))
        if not match:
            return None
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        month = months.get(match.group(1)[:3].lower())
        if not month:
            return None
        year = int(match.group(3))
        fiscal_year = year + (1 if month >= 4 else 0)
        quarter = ((month - 4) % 12) // 3 + 1
        return f"Q{quarter} FY{fiscal_year % 100:02d}"

    @classmethod
    def _deduplicate_metrics(cls, metrics: list[dict]) -> list[dict]:
        unique = {}
        for item in metrics:
            key = (item["metric"], item["period"], item["basis"])
            current = unique.get(key)
            item_source_periods = _periods(
                Path(item["source_file"]).stem.replace("_", " ")
            )
            current_source_periods = (
                _periods(Path(current["source_file"]).stem.replace("_", " "))
                if current
                else []
            )
            item_is_primary = item["period"] in item_source_periods
            current_is_primary = (
                current is not None and current["period"] in current_source_periods
            )
            if (
                current is None
                or (current["value"] is None and item["value"] is not None)
                or (
                    current["value"] is not None
                    and item["value"] is not None
                    and item_is_primary
                    and not current_is_primary
                )
            ):
                unique[key] = item
        return list(unique.values())

    @staticmethod
    def _quarterly_changes(metrics: list[dict]) -> list[dict]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for item in metrics:
            if item["value"] is not None and not item["review_required"]:
                groups[
                    (
                        item["metric"],
                        item["basis"],
                        item["currency"],
                        item["unit"],
                    )
                ].append(item)
        changes = []
        for (metric, basis, currency, unit), entries in groups.items():
            by_period = {
                item["period"]: item
                for item in sorted(entries, key=lambda row: _sort_period(row["period"]))
            }
            ordered = list(by_period.values())
            for previous, current in zip(ordered, ordered[1:]):
                if (
                    not FinancialDataExtractorTool._consecutive_quarters(
                        previous["period"], current["period"]
                    )
                    or previous["value"] == 0
                ):
                    continue
                change = {
                    "metric": metric,
                    "from_period": previous["period"],
                    "to_period": current["period"],
                    "basis": basis,
                    "currency": currency,
                    "unit": unit,
                    "previous_value": previous["value"],
                    "current_value": current["value"],
                }
                if metric == "operating_margin":
                    change["change_percentage_points"] = round(
                        current["value"] - previous["value"], 2
                    )
                else:
                    change["growth_pct"] = round(
                        (current["value"] - previous["value"])
                        / abs(previous["value"]) * 100,
                        2,
                    )
                changes.append(change)
        return changes

    @staticmethod
    def _consecutive_quarters(previous: str, current: str) -> bool:
        previous_year, previous_quarter = _sort_period(previous)
        current_year, current_quarter = _sort_period(current)
        expected = (
            previous_year + (previous_quarter == 4),
            1 if previous_quarter == 4 else previous_quarter + 1,
        )
        return (current_year, current_quarter) == expected

    @staticmethod
    def _matching_metric(line: str) -> str | None:
        for name, patterns in METRIC_PATTERNS.items():
            if any(pattern.search(line) for pattern in patterns):
                return name
        return None

    @staticmethod
    def _number(token: str) -> float:
        value = token.strip().rstrip("%").replace(",", "")
        if value.startswith("(") and value.endswith(")"):
            return -float(value[1:-1])
        return float(value)

    @classmethod
    def _number_or_none(cls, token: str) -> float | None:
        normalized = token.strip()
        if not normalized or normalized in {"-", "—", "–", "NA", "N/A"}:
            return None
        if normalized.startswith("(") and normalized.endswith(")"):
            return cls._number(normalized)
        match = NUMBER_PATTERN.search(normalized)
        if not match:
            return None
        return cls._number(match.group(0))


class QualitativeAnalysisTool:
    """Embedding-based semantic RAG, with explicit BM25-only offline fallback."""

    def __init__(
        self,
        data_dir: Path,
        top_k: int = 3,
        *,
        embeddings: Embeddings | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.repository = SourceRepository(data_dir)
        self.top_k = top_k
        self.embeddings = embeddings
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model

    def retrieve(self, query: str) -> dict:
        all_transcripts = [
            doc
            for doc in self.repository.load({"transcript"})
            if doc.doc_type == "transcript"
        ]
        by_file: dict[str, list[SourceDocument]] = defaultdict(list)
        for doc in all_transcripts:
            by_file[doc.source_file].append(doc)
        files = sorted(by_file, key=lambda name: _sort_period(
            by_file[name][0].period or name
        ))
        selected_names = files[-3:]
        selected = [doc for name in selected_names for doc in by_file[name]]
        chunks = self._chunks(selected)
        shared_terms = self._shared_terms(by_file, selected_names)
        topics = {
            "management_outlook": (
                "management outlook guidance forward-looking statements "
                f"future demand and next quarter {query}"
            ),
            "recurring_theme": (
                "demand trends client spending growth margins pricing deal wins "
                f"themes discussed across quarters {query}"
            ),
            "risk": (
                "business risks headwinds uncertainty client spending margin "
                f"pressure and challenges {query}"
            ),
            "opportunity": (
                "growth opportunities deal pipeline large deals artificial "
                f"intelligence cloud and client demand {query}"
            ),
        }
        vector_store = self._vector_store(chunks) if chunks else None
        retrieved = []
        for category, category_query in topics.items():
            ranked = (
                self._semantic_results(vector_store, chunks, category_query)
                if vector_store is not None
                else self._bm25(chunks, category_query)
            )
            ranked = self._diverse_results(ranked)[: self.top_k]
            for score, chunk in ranked:
                if score <= 0:
                    continue
                retrieved.append(
                    {
                        "category": category,
                        "source_file": chunk["source_file"],
                        "source_url": chunk["source_url"],
                        "period": chunk["period"],
                        "page": chunk["page"],
                        "section": chunk["section"],
                        "excerpt": chunk["text"][:1000],
                        "relevance_score": round(score, 4),
                        "retrieval_method": (
                            "semantic_embedding" if vector_store is not None else "BM25"
                        ),
                    }
                )
        retrieved.extend(
            self._recurring_theme_findings(chunks, shared_terms, query)
        )
        retrieved = self._deduplicate(retrieved)
        limitations = []
        if len(selected_names) < 2:
            limitations.append(
                "Fewer than two actual earnings-call transcripts were supplied; "
                "recurring themes cannot be reliably established."
            )
        if not selected_names:
            limitations.append(
                "No verified earnings-call transcripts were found. Add 2–3 actual "
                "transcripts and record their source URLs in source_manifest.json."
            )
        if any(doc.source_url is None for doc in selected):
            limitations.append(
                "Transcript excerpts without a source URL are local, unverified "
                "documents and should not be treated as authenticated quotations."
            )
        if not shared_terms and len(selected_names) >= 2:
            limitations.append(
                "No meaningful lexical themes were found in more than one supplied "
                "transcript; recurring themes are not reported."
            )
        return {
            "retrieval_method": (
                "LangChain InMemoryVectorStore semantic embedding retrieval"
                if vector_store is not None
                else "BM25 over overlapping transcript text chunks (offline mode)"
            ),
            "transcripts_used": selected_names,
            "recurring_terms": [
                {"term": term, "transcript_count": count}
                for term, count in shared_terms
            ],
            "findings": retrieved,
            "limitations": list(dict.fromkeys(limitations)),
        }

    def _vector_store(self, chunks: list[dict]) -> InMemoryVectorStore | None:
        embeddings = self.embeddings
        if embeddings is None and self.embedding_api_key:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.embedding_api_key,
            )
        if embeddings is None:
            return None
        self.embeddings = embeddings
        return InMemoryVectorStore.from_texts(
            [chunk["text"] for chunk in chunks],
            embedding=embeddings,
            metadatas=[
                {
                    "source_file": chunk["source_file"],
                    "source_url": chunk["source_url"] or "",
                    "period": chunk["period"] or "",
                    "page": chunk["page"],
                    "section": chunk["section"],
                }
                for chunk in chunks
            ],
        )

    def _semantic_results(
        self,
        vector_store: InMemoryVectorStore,
        chunks: list[dict],
        query: str,
    ) -> list[tuple[float, dict]]:
        by_identity = {
            (
                chunk["source_file"],
                chunk["page"],
                chunk["text"],
            ): chunk
            for chunk in chunks
        }
        results = []
        for document, score in vector_store.similarity_search_with_score(
            query, k=min(len(chunks), self.top_k * len({c["source_file"] for c in chunks}))
        ):
            metadata = document.metadata
            identity = (metadata["source_file"], metadata["page"], document.page_content)
            chunk = by_identity.get(identity)
            if chunk is not None:
                results.append((float(score), chunk))
        return results

    @staticmethod
    def _diverse_results(ranked: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
        by_file = {}
        for score, chunk in ranked:
            previous = by_file.get(chunk["source_file"])
            if previous is None or score > previous[0]:
                by_file[chunk["source_file"]] = (score, chunk)
        diversified = sorted(by_file.values(), key=lambda item: (-item[0], item[1]["source_file"]))
        seen = {chunk["source_file"] for _, chunk in diversified}
        diversified.extend(
            (score, chunk)
            for score, chunk in ranked
            if chunk["source_file"] not in seen
        )
        return diversified

    @classmethod
    def _shared_terms(
        cls, documents_by_file: dict[str, list[SourceDocument]], selected_names: list[str]
    ) -> list[tuple[str, int]]:
        term_counts = Counter()
        for file_name in selected_names:
            terms = {
                term
                for document in documents_by_file[file_name]
                for term in cls._tokens(document.text)
                if len(term) >= 3 and term not in RAG_STOPWORDS and not term.isdigit()
            }
            term_counts.update(terms)
        return sorted(
            (
                (term, count)
                for term, count in term_counts.items()
                if count >= 2
            ),
            key=lambda item: (-item[1], item[0]),
        )

    @classmethod
    def _recurring_theme_findings(
        cls,
        chunks: list[dict],
        shared_terms: list[tuple[str, int]],
        query: str,
    ) -> list[dict]:
        shared = {term for term, _ in shared_terms}
        if not shared:
            return []
        query_terms = set(cls._tokens(query))
        best_by_file: dict[str, tuple[float, dict, list[str]]] = {}
        for chunk in chunks:
            terms = set(cls._tokens(chunk["text"]))
            matched = sorted(terms & shared)
            if not matched:
                continue
            relevance = len(terms & query_terms)
            score = float(len(matched) + relevance * 0.1)
            previous = best_by_file.get(chunk["source_file"])
            if previous is None or score > previous[0]:
                best_by_file[chunk["source_file"]] = (score, chunk, matched)

        results = []
        for score, chunk, matched in sorted(
            best_by_file.values(),
            key=lambda item: (-item[0], item[1]["source_file"]),
        )[:3]:
            results.append(
                {
                    "category": "recurring_theme",
                    "source_file": chunk["source_file"],
                    "source_url": chunk["source_url"],
                    "period": chunk["period"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "excerpt": chunk["text"][:1000],
                    "recurring_terms": matched,
                    "relevance_score": round(score, 4),
                }
            )
        return results

    @staticmethod
    def _chunks(documents: list[SourceDocument]) -> list[dict]:
        chunks = []
        for doc in documents:
            paragraphs = [part.strip() for part in doc.text.splitlines() if part.strip()]
            buffer = ""
            for paragraph in paragraphs:
                if buffer and len(buffer) + len(paragraph) > 1400:
                    chunks.append(QualitativeAnalysisTool._chunk_record(doc, buffer))
                    overlap = buffer[-180:]
                    boundary = overlap.find(" ")
                    buffer = overlap[boundary + 1 :] if boundary >= 0 else overlap
                buffer = f"{buffer}\n{paragraph}".strip()
            if buffer:
                chunks.append(QualitativeAnalysisTool._chunk_record(doc, buffer))
        return chunks

    @staticmethod
    def _chunk_record(doc: SourceDocument, text: str) -> dict:
        lowered = text.lower()
        if re.search(r"\bq\s*&\s*a\b|question and answer", lowered):
            section = "Q&A"
        elif "outlook" in lowered or "forward-looking" in lowered:
            section = "Outlook"
        else:
            section = "Transcript"
        return {
            "text": text,
            "source_file": doc.source_file,
            "source_url": doc.source_url,
            "period": doc.period,
            "page": doc.page,
            "section": section,
        }

    @classmethod
    def _bm25(cls, chunks: list[dict], query: str) -> list[tuple[float, dict]]:
        tokenized = [cls._tokens(chunk["text"]) for chunk in chunks]
        query_terms = cls._tokens(query)
        if not tokenized or not query_terms:
            return []
        doc_freq = Counter(term for doc in tokenized for term in set(doc))
        average_length = sum(map(len, tokenized)) / len(tokenized) or 1
        scores = []
        k1, b = 1.5, 0.75
        for chunk, tokens in zip(chunks, tokenized):
            counts = Counter(tokens)
            score = 0.0
            for term in set(query_terms):
                frequency = counts[term]
                if not frequency:
                    continue
                inverse = math.log(
                    1 + (len(tokenized) - doc_freq[term] + 0.5)
                    / (doc_freq[term] + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * len(tokens) / average_length
                )
                score += inverse * frequency * (k1 + 1) / denominator
            scores.append((score, chunk))
        return sorted(scores, key=lambda item: item[0], reverse=True)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _deduplicate(items: list[dict]) -> list[dict]:
        positions = {}
        result = []
        for item in items:
            key = (item["category"], item["source_file"], item["excerpt"])
            if key not in positions:
                positions[key] = len(result)
                result.append(item)
            else:
                existing = result[positions[key]]
                recurring_terms = set(existing.get("recurring_terms", []))
                recurring_terms.update(item.get("recurring_terms", []))
                if recurring_terms:
                    existing["recurring_terms"] = sorted(recurring_terms)
        return result


def build_agent_tools(
    extractor: FinancialDataExtractorTool,
    qualitative: QualitativeAnalysisTool,
) -> list[StructuredTool]:
    def financial_extract(query: str, quarters: int = 4) -> str:
        return json.dumps(extractor.extract(query, quarters), ensure_ascii=False)

    def qualitative_retrieve(query: str) -> str:
        return json.dumps(qualitative.retrieve(query), ensure_ascii=False)

    def calculate_growth(current: float, previous: float) -> str:
        if previous == 0:
            return json.dumps({"growth_pct": None, "reason": "Previous value is zero."})
        return json.dumps(
            {"current": current, "previous": previous,
             "growth_pct": round((current - previous) / previous * 100, 2)}
        )

    return [
        StructuredTool.from_function(
            financial_extract,
            name="FinancialDataExtractorTool",
            description=(
                "Extract TCS quarterly revenue, net profit, and operating margin "
                "with source excerpts, URL, period, basis, units, and quarter changes."
            ),
        ),
        StructuredTool.from_function(
            qualitative_retrieve,
            name="QualitativeAnalysisTool",
            description=(
                "Use semantic embeddings when configured, otherwise explicitly "
                "labeled BM25 retrieval, to find cited outlook, themes, risks and "
                "opportunities across up to three TCS call transcripts."
            ),
        ),
        StructuredTool.from_function(
            calculate_growth,
            name="financial_growth_calculator",
            description="Calculate percentage change between two verified actual values.",
        ),
    ]
