import json
from pathlib import Path

from langchain_core.embeddings import Embeddings
from app.sources import SourceRepository
from app.tools import FinancialDataExtractorTool, QualitativeAnalysisTool


def test_financial_extractor_maps_periods_and_preserves_evidence(tmp_path):
    (tmp_path / "tcs_q2fy25_results.txt").write_text(
        "CONSOLIDATED FINANCIAL RESULTS (INR Crore)\n"
        "Q2 FY25 Q1 FY25\n"
        "Revenue from Operations 64,259 62,613\n"
        "Net Income 11,909 11,603\n"
        "Operating Margin % 24.1% 24.0%\n",
        encoding="utf-8",
    )
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "tcs_q2fy25_results.txt": {
                    "doc_type": "financial_report",
                    "period": "Q2 FY25",
                    "source_url": "https://example.test/quarterly-results.pdf",
                    "basis": "consolidated",
                    "currency": "INR",
                    "unit": "crore",
                }
            }
        ),
        encoding="utf-8",
    )

    result = FinancialDataExtractorTool(tmp_path).extract("latest", quarters=2)
    by_metric_period = {(item["metric"], item["period"]): item for item in result["metrics"]}
    assert by_metric_period[("revenue", "Q2 FY25")]["value"] == 64259
    assert by_metric_period[("revenue", "Q1 FY25")]["value"] == 62613
    assert by_metric_period[("net_profit", "Q2 FY25")]["value"] == 11909
    assert by_metric_period[("operating_margin", "Q2 FY25")]["value"] == 24.1
    assert by_metric_period[("revenue", "Q2 FY25")]["source_url"].startswith("https://")
    assert not by_metric_period[("revenue", "Q2 FY25")]["review_required"]


def test_financial_extractor_flags_unmapped_multi_period_rows(tmp_path):
    (tmp_path / "quarterly_report.txt").write_text(
        "INR crore\nRevenue from Operations 64000 62000\n",
        encoding="utf-8",
    )
    (tmp_path / "source_manifest.json").write_text(
        '{"quarterly_report.txt":{"doc_type":"financial_report"}}',
        encoding="utf-8",
    )
    result = FinancialDataExtractorTool(tmp_path).extract()
    row = result["metrics"][0]
    assert row["value"] is None
    assert row["review_required"] is True
    assert row["period"] == "period_unresolved"


def test_qualitative_retrieval_returns_relevant_source_chunks(tmp_path):
    for quarter, text in {
        "q2": "Management outlook: cloud adoption and client demand remain cautious.",
        "q3": "Risk: macro uncertainty may delay spending; cloud modernization pipeline grew.",
    }.items():
        name = f"tcs_{quarter}_transcript.txt"
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                f"tcs_{quarter}_transcript.txt": {
                    "doc_type": "transcript",
                    "period": period,
                    "source_url": f"https://example.test/{quarter}",
                }
                for quarter, period in (("q2", "Q2 FY25"), ("q3", "Q3 FY25"))
            }
        ),
        encoding="utf-8",
    )

    result = QualitativeAnalysisTool(tmp_path).retrieve("cloud opportunity and risk")
    assert result["retrieval_method"].startswith("BM25")
    assert set(result["transcripts_used"]) == {
        "tcs_q2_transcript.txt",
        "tcs_q3_transcript.txt",
    }
    assert result["findings"]
    assert all(item["source_url"].startswith("https://") for item in result["findings"])


def test_retrieval_is_empty_and_explicit_when_transcripts_absent(tmp_path):
    result = QualitativeAnalysisTool(tmp_path).retrieve("outlook")
    assert result["findings"] == []
    assert any("No verified" in item for item in result["limitations"])


def test_recurring_themes_require_cross_transcript_evidence(tmp_path):
    (tmp_path / "q2_transcript.txt").write_text(
        "Clients are expanding cloud modernization demand. Oneoff project was announced.",
        encoding="utf-8",
    )
    (tmp_path / "q3_transcript.txt").write_text(
        "Cloud modernization demand increased for clients. Separate initiative launched.",
        encoding="utf-8",
    )
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "q2_transcript.txt": {
                    "doc_type": "transcript",
                    "period": "Q2 FY25",
                    "source_url": "https://example.test/q2",
                },
                "q3_transcript.txt": {
                    "doc_type": "transcript",
                    "period": "Q3 FY25",
                    "source_url": "https://example.test/q3",
                },
            }
        ),
        encoding="utf-8",
    )

    result = QualitativeAnalysisTool(tmp_path).retrieve("cloud demand")
    recurrent_terms = {item["term"] for item in result["recurring_terms"]}
    recurring_files = {
        item["source_file"]
        for item in result["findings"]
        if item["category"] == "recurring_theme"
    }

    assert {"cloud", "modernization", "demand"} <= recurrent_terms
    assert "oneoff" not in recurrent_terms
    assert recurring_files == {"q2_transcript.txt", "q3_transcript.txt"}
    assert all(
        item["recurring_terms"]
        for item in result["findings"]
        if item["category"] == "recurring_theme"
    )


def test_rag_flags_transcripts_with_unrecorded_source_urls(tmp_path):
    for name in ("q2_transcript.txt", "q3_transcript.txt"):
        (tmp_path / name).write_text(
            "Cloud demand and client projects are discussed.",
            encoding="utf-8",
        )
    result = QualitativeAnalysisTool(tmp_path).retrieve("cloud demand")
    assert any("without a source URL" in item for item in result["limitations"])


def test_actual_tcs_quarterly_pdfs_extract_cited_values_and_comparisons():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    result = FinancialDataExtractorTool(data_dir).extract(quarters=3)
    indexed = {
        (item["period"], item["metric"]): item
        for item in result["metrics"]
        if item["basis"].startswith("consolidated")
    }

    expected = {
        ("Q4 FY26", "revenue"): (70698.0, "financial_reports/Q4_FY26_results.pdf"),
        ("Q4 FY26", "net_profit"): (13784.0, "financial_reports/Q4_FY26_results.pdf"),
        ("Q4 FY26", "operating_margin"): (
            25.3,
            "financial_summaries/Q4_FY26_press_release.pdf",
        ),
        ("Q1 FY27", "revenue"): (72275.0, "financial_reports/Q1_FY27_results.pdf"),
        ("Q1 FY27", "net_profit"): (13420.0, "financial_reports/Q1_FY27_results.pdf"),
        ("Q1 FY27", "operating_margin"): (
            24.0,
            "financial_summaries/Q1_FY27_press_release.pdf",
        ),
        ("Q2 FY27", "revenue"): (73188.0, "financial_reports/Q2_FY27_results.pdf"),
        ("Q2 FY27", "net_profit"): (13934.0, "financial_reports/Q2_FY27_results.pdf"),
        ("Q2 FY27", "operating_margin"): (
            24.0,
            "financial_summaries/Q2_FY27_press_release.pdf",
        ),
    }
    assert set(indexed) == set(expected)
    for key, (value, source_file) in expected.items():
        metric = indexed[key]
        assert metric["value"] == value
        assert metric["source_file"] == source_file
        assert metric["source_url"].startswith("https://")
        assert metric["page"] is not None
        assert metric["review_required"] is False
    assert indexed[("Q1 FY27", "operating_margin")]["basis"] == (
        "consolidated (excludes exceptional items)"
    )
    revenue_changes = [
        item for item in result["quarter_over_quarter"]
        if item["metric"] == "revenue"
    ]
    assert [
        (item["from_period"], item["to_period"], item["growth_pct"])
        for item in revenue_changes
    ] == [
        ("Q4 FY26", "Q1 FY27", 2.23),
        ("Q1 FY27", "Q2 FY27", 1.26),
    ]
    assert not any(
        item["metric"] == "operating_margin"
        for item in result["quarter_over_quarter"]
    )


def test_pdf_table_number_parser_preserves_negatives_and_missing_values():
    assert FinancialDataExtractorTool._number_or_none("(1,234)") == -1234
    assert FinancialDataExtractorTool._number_or_none("-") is None
    assert FinancialDataExtractorTool._number_or_none("N/A") is None


def test_text_fallback_preserves_negative_and_missing_metrics(tmp_path):
    (tmp_path / "Q1_FY26_results.txt").write_text(
        "CONSOLIDATED FINANCIAL RESULTS (INR crore)\n"
        "Q1 FY26\n"
        "Revenue from operations -\n"
        "Profit for the period (1,234)\n",
        encoding="utf-8",
    )
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "Q1_FY26_results.txt": {
                    "doc_type": "financial_report",
                    "period": "Q1 FY26",
                    "basis": "consolidated",
                    "currency": "INR",
                    "unit": "crore",
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = FinancialDataExtractorTool(tmp_path).extract()["metrics"]
    by_name = {item["metric"]: item for item in metrics}
    assert by_name["net_profit"]["value"] == -1234
    assert by_name["net_profit"]["unit"] == "crore"
    assert by_name["revenue"]["value"] is None
    assert by_name["revenue"]["review_required"] is True


class SynonymEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        lowered = text.lower()
        if (
            "unit economics deteriorated" in lowered
            or "margin pressure" in lowered
        ):
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_semantic_rag_finds_paraphrased_evidence_with_citation_metadata(tmp_path):
    documents = {
        "q2_call.txt": "Unit economics deteriorated after borrowing costs rose.",
        "q3_call.txt": "The call began with introductions and housekeeping.",
    }
    for name, text in documents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "q2_call.txt": {
                    "doc_type": "transcript",
                    "period": "Q2 FY25",
                    "source_url": "https://example.test/q2",
                },
                "q3_call.txt": {
                    "doc_type": "transcript",
                    "period": "Q3 FY25",
                    "source_url": "https://example.test/q3",
                },
            }
        ),
        encoding="utf-8",
    )

    result = QualitativeAnalysisTool(
        tmp_path, embeddings=SynonymEmbeddings()
    ).retrieve("pricing squeeze")
    risk_findings = [
        item for item in result["findings"] if item["category"] == "risk"
    ]

    assert "InMemoryVectorStore" in result["retrieval_method"]
    assert any("Unit economics deteriorated" in item["excerpt"] for item in risk_findings)
    assert all(item["source_url"].startswith("https://") for item in risk_findings)
    assert all(item["period"] == "Q2 FY25" for item in risk_findings)
    assert all(item["section"] for item in risk_findings)


def test_actual_transcript_pdfs_load_with_manifest_citations():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    result = QualitativeAnalysisTool(data_dir).retrieve("AI opportunities and demand")
    assert result["retrieval_method"].startswith("BM25")
    assert len(result["transcripts_used"]) == 3
    assert result["findings"]
    assert all(
        item["source_url"].startswith("https://") and item["page"] is not None
        for item in result["findings"]
    )


def test_source_manifest_matches_nine_downloaded_readable_pdfs():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    documents = SourceRepository(data_dir).load()
    files = {item.source_file for item in documents}
    expected = {
        "financial_reports/Q4_FY26_results.pdf",
        "financial_reports/Q1_FY27_results.pdf",
        "financial_reports/Q2_FY27_results.pdf",
        "financial_summaries/Q4_FY26_press_release.pdf",
        "financial_summaries/Q1_FY27_press_release.pdf",
        "financial_summaries/Q2_FY27_press_release.pdf",
        "earnings_call_transcripts/Q4_FY26_transcript.pdf",
        "earnings_call_transcripts/Q1_FY27_transcript.pdf",
        "earnings_call_transcripts/Q2_FY27_transcript.pdf",
    }

    assert expected <= files
    for relative in expected:
        page = next(item for item in documents if item.source_file == relative)
        assert page.text.strip()
        assert page.page is not None
        assert page.source_url.startswith("https://")
        assert page.publication_date
        assert page.retrieval_status == "downloaded_pdf_text_verified"
