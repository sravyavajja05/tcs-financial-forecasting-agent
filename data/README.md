# Source documents and provenance

This Version 2 data directory contains the nine actual TCS quarterly filings
listed in [`source_manifest.json`](source_manifest.json), downloaded and text
verified on 2026-10-27. The results are TCS filings hosted in BSE's archive;
the press releases and earnings-call transcripts are TCS filings hosted in
NSE's archive. Every manifest entry records its exact source URL, period,
publication/filing date, local filename, retrieval date, and retrieval status.

## Included source set

- Q4 FY26, Q1 FY27, and Q2 FY27 audited quarterly financial results in
  `financial_reports/`.
- The corresponding TCS quarterly press releases in `financial_summaries/`.
- The corresponding earnings-call transcripts in
  `earnings_call_transcripts/`.

## Adding or replacing a source

1. Save the original source PDF/TXT/Markdown document in the appropriate
   subdirectory.
2. Add a manifest entry keyed by its relative path. Record an exact, public
   source URL and publication date; do not use a general investor-relations
   landing page as the URL for a specific document.
3. Use `source_manifest.example.json` as a schema example only. Replace its
   sample filenames and dates; do not leave placeholder metadata in the live
   manifest.
4. Confirm the document is readable and verify financial figures and transcript
   excerpts in the original before relying on them.

Financial statement tables are parsed with `pdfplumber`; text fallback handles
table layouts whose extracted cells do not align. Pages without selectable PDF
text are not OCR'd automatically and must be processed and verified separately.
Standalone results are not merged with consolidated results. Quarter mapping
uses the dated “Three months ended” columns, not six-month or annual totals.
Ambiguous or missing values remain null and require review.

Transcript retrieval is local and source-grounded: real mode uses an
embedding-backed LangChain vector store; offline mock mode uses BM25 and labels
that retrieval method honestly. The application does not fetch or fabricate
documents at request time.
