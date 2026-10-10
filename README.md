# AI Data Analyst

Ask questions about a CSV/Excel file in plain English. An LLM (Gemini or Claude) writes pandas code, the code runs in a sandboxed subprocess, and you get a table or chart, a short explanation, and the code that produced it.

## Architecture

```
Question ──> schema prompt ──> LLM (Gemini or Claude) writes pandas code ──> AST safety check
   ^        (columns, types,        (Gemini / Claude)            │
   │         value hints,                                         v
   │         3 sample rows)                         sandboxed subprocess (timeout,
   │                                                 memory limit, no API key in env)
   │                                                              │
   │               error? ── send traceback back to the LLM ──────┤ (up to 2 retries)
   │                                                              v
   └──── chat UI (Streamlit) <── explanation + table/chart <── result
```

- `llm.py`: one `chat()` interface over Gemini and Claude
- `analyst.py`: prompt building, retry loop, explanation
- `sandbox.py` + `sandbox_runner.py`: validation and isolated execution
- `app.py`: Streamlit chat UI (shows generated code for every answer)
- `make_sample_data.py`, `build_benchmark.py`, `eval.py`: dataset, ground-truth benchmark, scorer

## Run it

Get a free Gemini API key at https://aistudio.google.com (no credit card needed), then:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key         # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
python make_sample_data.py             # creates data/sales.csv
streamlit run app.py
```

Using Claude instead? Set `ANTHROPIC_API_KEY` (or pick Claude in the sidebar). Models can be overridden with `GEMINI_MODEL` / `CLAUDE_MODEL` (default Gemini model: `gemini-3.8-flash`; the sidebar also has a Model box).

Click **Use sample sales data** or upload your own file. Try: "Which region has the highest revenue?", "Plot monthly revenue", then a follow-up like "Now break that down by category".

Free-tier note: Gemini's free tier is rate limited, so a full benchmark run can pause and retry on 429 errors. That is expected.

## Evaluate it (this is the part worth putting on your resume)

```bash
python build_benchmark.py              # 20 questions, ground truth computed with pandas
python eval.py                         # full pipeline
python eval.py --retries 0             # ablation: no self-correction
python eval.py --basic-schema          # ablation: dtypes only, no value hints
python eval.py --provider claude       # same benchmark, different model
```

`eval.py` prints overall and first-try accuracy and saves every generated program to `eval/results.json`. Read the failures: they tell you what to fix in the prompt, and they make good interview stories.

Keep a table of your own runs here:

| Config | Accuracy |
|---|---|
| Gemini, basic schema, no retries | _run it_ |
| Gemini, rich schema, no retries | _run it_ |
| Gemini, rich schema, 2 retries | _run it_ |
| Claude, rich schema, 2 retries (optional) | _run it_ |

## Privacy and safety

- Only the schema, a few value hints and 3 sample rows are sent to the LLM provider (on Gemini's free tier, Google may use prompts to improve its products, so don't use sensitive data), never the full dataset.
- Generated code is validated (allowed imports only, no file/network helpers, no dunder access), then run in a separate `python -I` process with a timeout, CPU/memory limits (Linux/macOS), and an environment that does not include your API key.
- **This is a best-effort sandbox, not a security boundary.** For a public deployment, run the subprocess in Docker with no network and a read-only filesystem, or use a hosted sandbox such as E2B.
- LLMs can be confidently wrong. That is why every answer shows its code, and why the benchmark exists.

## Ideas to extend it

- Text-to-SQL with DuckDB and multi-table joins
- A data dictionary (column meanings, business rules) passed into the prompt
- Auto-EDA on upload: missing values, outliers, correlations
- Tool calling instead of free-form code
- Swap the sample data for a real dataset such as VTU results or IPL stats, and add 10 harder questions

## Troubleshooting

- **404 NOT_FOUND / model no longer available:** Google retires models regularly. Type the current model name into the sidebar Model box, or set `GEMINI_MODEL` (check aistudio.google.com for the current Flash model).
- **429 errors:** free-tier rate limit; the app waits and retries automatically.
- **PowerShell:** use `$env:GEMINI_API_KEY="your_key"` instead of `export`.
