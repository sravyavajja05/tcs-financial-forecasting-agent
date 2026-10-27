# TCS Financial Forecasting Agent — Version 2

Version 2 is a separate, source-grounded forecasting service. It does not
modify Version 1. Its workflow is:

1. Validate a `POST /forecast` request with Pydantic.
2. Log the request and execution status through SQLAlchemy.
3. Extract quarterly metrics and cited transcript passages from the bundled
   documents.
4. Calculate conservative, mechanical quarter-ahead estimates from comparable
   confirmed observations.
5. In real mode, use a LangChain agent and configured OpenAI models to write a
   concise qualitative assessment from the supplied evidence.
6. Validate the structured response and persist its JSON before returning it.

The service never downloads documents at request time and does not invent
missing figures, source references, or quotations.

## What is implemented

- **FastAPI:** `POST /forecast`, `GET /health`, and
  `GET /logs/{request_id}`.
- **Structured output:** `ForecastRequest` and `ForecastResponse` Pydantic
  models reject invalid requests and validate the final JSON response.
- **`FinancialDataExtractorTool`:** uses `pdfplumber` table extraction and
  extracted PDF text, with PyMuPDF text fallback for pages without readable
  `pdfplumber` text. It retains period, consolidated/standalone basis, unit,
  currency, source URL, page, and evidence excerpt. It maps only columns under
  “Three months ended” for quarter results; six-month and full-year totals are
  not treated as quarterly figures. Negative values and dashes/missing values
  are handled explicitly. Quarter-over-quarter changes are calculated only for
  consecutive quarters with matching basis and units.
- **`QualitativeAnalysisTool`:** chunks transcript PDF text with overlap and
  keeps file, URL, quarter, page, and section metadata. Real mode uses
  `OpenAIEmbeddings` and LangChain `InMemoryVectorStore` for semantic retrieval.
  Offline mode uses the existing BM25 retriever and labels it as BM25; it does
  not claim that lexical matching is semantic retrieval. Exact recurring terms
  are labeled as lexical recurrence, not proof of semantic recurrence.
- **LangChain agent:** the real-mode workflow registers both specialized tools
  and a growth calculator. A configured chat model synthesizes management
  outlook, recurring themes, risks, and opportunities from retrieved evidence.
  The model is not allowed to replace deterministic historical values,
  citations, period, or numerical estimates in the API response.
- **Persistence:** each request, execution status, and final response JSON are
  persisted in MySQL 8.0. SQLite remains available for local demos and tests.
- **Modes:** `MOCK_MODE=true` is offline and returns `generation_mode: "mock"`;
  it does not call an LLM or embeddings service. `MOCK_MODE=false` requires
  OpenAI credentials and uses real provider calls; the response is labeled
  `generation_mode: "real_ai"`.

## Verified TCS source documents

Nine PDFs were downloaded on **2026-10-27**, with extractable text confirmed.
The source set covers Q4 FY26, Q1 FY27, and Q2 FY27. The quarterly financial
statements are TCS filings hosted in BSE's corporate filing archive. The
earnings-call transcripts and press releases are TCS filings hosted in NSE's
corporate archive. Direct TCS site access was not required for these specific
source URLs. `data/source_manifest.json` records each local filename, source
URL, publication/filing date, retrieval date, document type, and retrieval
status.

| Quarter | Document | Publication / filing date | Local file | Exact source |
|---|---|---:|---|---|
| Q2 FY27 | Audited financial results | 2026-10-08 | `data/financial_reports/Q2_FY27_results.pdf` | [BSE filing](https://www.bseindia.com/xml-data/corpfiling/AttachHis/2b021530-bbc2-4590-8375-cefc205a7132.pdf) |
| Q2 FY27 | TCS press release | 2026-10-08 | `data/financial_summaries/Q2_FY27_press_release.pdf` | [NSE filing](https://nsearchives.nseindia.com/corporate/TCS_CORPCS_08102026154325_PressReleaseletter.pdf) |
| Q2 FY27 | Earnings-call transcript (call 2026-10-08) | 2026-10-14 | `data/earnings_call_transcripts/Q2_FY27_transcript.pdf` | [NSE filing](https://nsearchives.nseindia.com/corporate/TCS_CORPCS_14102026155941_SEInt14102026_signed.pdf) |
| Q1 FY27 | Audited financial results | 2026-07-09 | `data/financial_reports/Q1_FY27_results.pdf` | [BSE filing](https://www.bseindia.com/xml-data/corpfiling/AttachHis/ef21b97e-798a-47fa-9dd6-cd22d1fc758a.pdf) |
| Q1 FY27 | TCS press release | 2026-07-09 | `data/financial_summaries/Q1_FY27_press_release.pdf` | [NSE filing](https://nsearchives.nseindia.com/corporate/TCS_CORPCS_09072026155553_Press_release_signed.pdf) |
| Q1 FY27 | Earnings-call transcript (call 2026-07-09) | 2026-07-15 | `data/earnings_call_transcripts/Q1_FY27_transcript.pdf` | [NSE filing](https://nsearchives.nseindia.com/corporate/TCS_CORPCS_15072026193646_SEInt15072026_Signed.pdf) |
| Q4 FY26 | Audited financial results | 2026-04-09 | `data/financial_reports/Q4_FY26_results.pdf` | [BSE filing](https://www.bseindia.com/xml-data/corpfiling/AttachHis/da445746-06b9-4078-a71b-cc48f97fbc93.pdf) |
| Q4 FY26 | TCS press release | 2026-04-09 | `data/financial_summaries/Q4_FY26_press_release.pdf` | [NSE filing](https://nsearchives.nseindia.com/corporate/TCS_CORPCS_09042026155806_PressReleaseletter_signed.pdf) |
| Q4 FY26 | Earnings-call transcript (call 2026-04-09) | 2026-04-14 | `data/earnings_call_transcripts/Q4_FY26_transcript.pdf` | [NSE filing](https://nsearchives.nseindia.com/corporate/TCS_CORPCS_14042026200132_SEInt14042026_signed.pdf) |

Financial statements report “Profit for the period/year”; the extractor maps
that audited statement row to the API's `net_profit` metric and preserves that
row as the evidence excerpt. Press-release “net income” or “net profit” values
can differ because attribution and exceptional-item treatment differ. The
extractor does not silently substitute one definition for another. Operating
margin comes only from the corresponding TCS press release; it is not inferred.
The Q1 FY27 press release marks its operating-margin figure with an asterisk,
so that source's footnote and treatment should be checked when comparing it
with other periods.

## AI stack, reasoning, and guardrails

- **LLM provider:** OpenAI through `langchain-openai`; default chat model
  `gpt-4o-mini`, configurable with `LLM_MODEL`.
- **Embedding model:** OpenAI `text-embedding-3-small` by default, configurable
  with `OPENAI_EMBEDDING_MODEL`.
- **Vector store:** LangChain `InMemoryVectorStore`; embeddings and index are
  in-process and are rebuilt from local transcripts for each forecast.
- **Retrieval:** transcript pages are broken into overlapping text chunks.
  Topic queries target management outlook/forward-looking statements, demand
  and margin themes, risks, and opportunities such as deal pipeline and AI.
  Retrieved excerpts retain source URL, quarter, page, and section. Retrieval
  covers up to the latest three supplied transcript periods.
- **Reasoning:** financial metrics are extracted and trend estimates calculated
  before narrative generation. The LLM receives the deterministic results and
  retrieved excerpts, and returns prose in the existing Pydantic
  `ForecastResponse` schema. Estimates are explicitly mechanical trend
  extrapolations, not management guidance.
- **Grounding:** prompts prohibit inventing figures, quotations, source URLs, or
  quarters; document content is treated as untrusted evidence. Historical
  values, estimates, source citations, forecast period, and mode in the final
  response are set from deterministic application data rather than copied from
  model output. Hidden chain-of-thought is not requested or returned.
- **Retries:** a single bounded retry is used for recognized transient provider
  failures and for invalid/missing structured responses. Other provider errors
  fail visibly instead of falling back to a success-shaped mock response.
- **Evaluation:** tests cover the actual packaged PDFs, metric values and
  quarter comparisons, negative/missing values, citations, a paraphrased query
  against a deterministic fake embedding implementation, structured output,
  retry bounds, API logging, and SQLite persistence. The semantic test verifies
  the vector-store path; it does not make an external embedding request.

## Setup on Windows PowerShell

Run these commands from the Version 2 folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### MySQL 8.0

Create a database and least-privilege application user using MySQL Workbench
or the MySQL command-line client:

```sql
CREATE DATABASE tcs_forecasting_v2
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE USER 'forecast_user'@'localhost' IDENTIFIED BY 'REPLACE_WITH_A_STRONG_PASSWORD';
GRANT SELECT, INSERT, UPDATE, CREATE, ALTER, INDEX, REFERENCES
  ON tcs_forecasting_v2.* TO 'forecast_user'@'localhost';
```

Set `MYSQL_URL` in the local `.env`, using your own account password:

```dotenv
MYSQL_URL=mysql+pymysql://forecast_user:your_password@127.0.0.1:3306/tcs_forecasting_v2?charset=utf8mb4
```

`MYSQL_URL` is primary; `DATABASE_URL` remains a backward-compatible fallback.
If both are defined, `MYSQL_URL` wins. The application initializes its tables
on startup. For SQLite, the application creates the database's parent folder
if it is missing. Keep actual credentials in `.env`, never in `.env.example` or
source control. `.env.example` uses placeholders.

For an offline SQLite demo, set:

```dotenv
MOCK_MODE=true
MYSQL_URL=sqlite:///./storage/tcs_forecasting_v2.sqlite3
```

For real AI, set `MOCK_MODE=false`, provide `OPENAI_API_KEY`, and optionally
set `LLM_MODEL` and `OPENAI_EMBEDDING_MODEL`. Both embeddings and chat use the
configured OpenAI key. A provider/network failure is surfaced; real mode does
not silently downgrade to BM25 or mock generation.

### Run the service

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

The API is documented at `http://127.0.0.1:8000/docs`; check
`http://127.0.0.1:8000/health` before sending requests. If the configured
database is unavailable, the health endpoint reports the degraded state and
forecast requests fail rather than claiming persistence succeeded.

### Example request and response

```powershell
$body = @{
  query = "Assess TCS revenue, net profit, margin, outlook, risks and opportunities from supplied sources."
  quarters = 3
} | ConvertTo-Json

$forecast = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/forecast `
  -ContentType "application/json" `
  -Body $body

$forecast | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://127.0.0.1:8000/logs/$($forecast.request_id)"
```

Abbreviated offline response generated from the bundled sources (the estimate
is a deterministic trend calculation, not a live LLM forecast; the historical
metrics and citation arrays are shortened here for readability):

```json
{
  "request_id": "generated-uuid",
  "company": "Tata Consultancy Services",
  "forecast_period": "Q3 FY27",
  "generation_mode": "mock",
  "historical_metrics": [
    {
      "period": "Q2 FY27",
      "metric": "revenue",
      "value": 73188.0,
      "unit": "crore",
      "currency": "INR",
      "basis": "consolidated",
      "source_file": "financial_reports/Q2_FY27_results.pdf",
      "source_url": "https://www.bseindia.com/xml-data/corpfiling/AttachHis/2b021530-bbc2-4590-8375-cefc205a7132.pdf",
      "page": 8,
      "excerpt": "Revenue from operations: 73,188",
      "review_required": false
    }
  ],
  "estimate": {
    "revenue_next_quarter_inr_crore": 74466.54,
    "net_profit_next_quarter_inr_crore": 14016.86,
    "operating_margin_next_quarter_pct": 24.65,
    "method": "Mechanical historical trend only: average sequential change across up to the latest three confirmed quarters; margin uses the mean of the latest comparable reported margins. Not management guidance."
  },
  "forecast": {
    "revenue_outlook": "Mechanical trend estimate: 74,466.54 crore based on confirmed report values; this is not guidance.",
    "net_profit_outlook": "Mechanical trend estimate: 14,016.86 crore based on confirmed report values; this is not guidance.",
    "operating_margin_outlook": "Mechanical trend estimate: 24.43 % based on confirmed report values; this is not guidance.",
    "management_outlook": [],
    "recurring_themes": [],
    "risks": [],
    "opportunities": [],
    "confidence": "Low",
    "assumptions": []
  },
  "source_evidence": [
    {
      "category": "financial",
      "source_file": "financial_reports/Q2_FY27_results.pdf",
      "source_url": "https://www.bseindia.com/xml-data/corpfiling/AttachHis/2b021530-bbc2-4590-8375-cefc205a7132.pdf",
      "period": "Q2 FY27",
      "page": 8,
      "excerpt": "Revenue from operations: 73,188"
    }
  ],
  "limitations": [
    "The Q1 FY27 operating margin is explicitly adjusted to exclude exceptional items and is not treated as comparable to unadjusted quarterly margins.",
    "MOCK_MODE=true: no LLM was called."
  ],
  "disclaimer": "This is an analytical estimate, not investment advice or a guarantee of future performance."
}
```

The example omits additional historical metrics and retrieved transcript
evidence for readability. It reports the actual response values generated by
the offline calculation over Q4 FY26–Q2 FY27. The service marks offline output
as mock and never presents it as real AI output.

## Tests and verification

Run the complete test suite:

```powershell
python -m pytest -v
```

Tests use SQLite, temporary source fixtures, and a deterministic embedding
stub. They do not call a live LLM or embedding service. The MySQL
write/read integration test is opt-in; run it only against a disposable MySQL
8.0 test schema by setting `TCS_V2_TEST_DATABASE_URL` in the test process.
Without that URL, the live-MySQL integration test is skipped and MySQL must not
be reported as verified.

## Limitations

- No real OpenAI embedding or chat request was made as part of offline tests.
  Configure credentials and network access to verify real-provider output.
- Live MySQL connectivity and writes require a configured test URL and
  reachable MySQL 8.0 service; SQLite tests are not proof of MySQL integration.
- PDF text fallback is not OCR. Image-only/scanned documents need OCR and
  manual verification before adding them as evidence.
- The vector store is in-memory and rebuilt for each request; it is not a
  persistent vector database. This keeps local operation simple but adds
  embedding latency and API cost in real mode.
- Management outlook and risk interpretations depend on retrieval quality and
  should be checked against cited transcript passages. Lexical BM25 in mock
  mode cannot match paraphrases and is not semantic retrieval.
- Trend projections are simple descriptive arithmetic, not a statistical
  forecasting model, investment advice, or a representation of TCS guidance.

## Repository status

The workspace and Version 2 folder are not Git repositories, so there is no
remote URL or existing history to report. No repository was initialized and no
changes were committed or pushed. To publish later, create a repository under
your own account, initialize Git in the desired project folder, review the
included source files and PDFs, add a remote URL from that repository, then
commit and push only after checking that `.env`, databases, and credentials are
excluded.
