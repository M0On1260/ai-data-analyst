"""Core logic: question -> generated code -> sandboxed execution -> explanation."""
from __future__ import annotations

import re
import time

import pandas as pd

from llm import get_client  # noqa: F401  (re-exported for app.py / eval.py)
from sandbox import run_code

SYSTEM_PROMPT = """You are a senior data analyst. A pandas DataFrame named `df` is already loaded.
Write Python code that answers the user's question about it.

Rules:
- Already available: pd, np, px (plotly.express), go (plotly.graph_objects). You may also import math, statistics, datetime, re, collections, itertools, calendar.
- Never read or write files, use the network, or call input().
- Put the final answer in a variable named `result`: a plain scalar for single-value answers, or a DataFrame/Series for tables. Keep `result` minimal: exactly what was asked, nothing extra.
- If a chart is requested or clearly helpful, also assign a plotly figure to `fig`.
- Date columns may be stored as strings: convert with pd.to_datetime before using them.
- Do not modify `df` in place; work on a copy.
- Express percentages as numbers from 0 to 100.
- Return ONLY Python code in a single ```python block. No explanation."""

EXPLAIN_PROMPT = (
    "You explain data analysis results to a non-technical reader in 2-4 sentences. "
    "State the answer first. Mention a caveat only if it is visible in the code "
    "(for example missing values being dropped). Never invent numbers that are not in the result."
)


def build_schema(df: pd.DataFrame, rich: bool = True, max_unique: int = 8) -> str:
    """Describe the data for the LLM. Never sends more than a few sample rows."""
    if not rich:
        return f"Columns and dtypes:\n{df.dtypes.to_string()}"

    lines = [f"Rows: {len(df)}", "Columns:"]
    for col in df.columns:
        s = df[col]
        info = f"- {col} ({s.dtype}), {int(s.isna().sum())} missing"
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info += f", min={s.min()}, max={s.max()}"
        else:
            uniq = s.dropna().unique()
            info += f", {len(uniq)} unique"
            if 0 < len(uniq) <= max_unique:
                info += f": {list(uniq)}"
        lines.append(info)
    lines.append("\nSample rows:\n" + df.head(3).to_string())
    return "\n".join(lines)


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (match.group(1) if match else text).strip()


def _history_block(history) -> str:
    if not history:
        return ""
    parts = ["Earlier in this session (use it to resolve follow-up questions):"]
    for q, code in history[-3:]:
        parts.append(f"Q: {q}\nCode used:\n{code}")
    return "\n\n".join(parts) + "\n\n"


def _result_text(run: dict) -> str:
    if run["kind"] == "table":
        shown = run["table"].head(20).to_string(index=False)
        return f"{shown}\n({run['total_rows']} rows in total)"
    return run["text"]


def explain(client, question: str, code: str, run: dict) -> str:
    return client.chat(
        EXPLAIN_PROMPT,
        [{
            "role": "user",
            "content": f"Question: {question}\n\nCode:\n{code}\n\nResult:\n{_result_text(run)}",
        }],
        max_tokens=400,
    ).strip()


def analyze(
    df: pd.DataFrame,
    question: str,
    client,
    history=None,
    max_retries: int = 2,
    rich_schema: bool = True,
    explain_result: bool = True,
    timeout: int = 20,
    on_step=None,
) -> dict:
    """Answer one question. Retries with the error message if the code fails."""
    prompt = f"{build_schema(df, rich_schema)}\n\n{_history_block(history)}Question: {question}"
    messages = [{"role": "user", "content": prompt}]
    step = on_step or (lambda text: None)
    started = time.time()

    code, run, attempts = "", {"ok": False, "error": "not run"}, 0
    for _ in range(max_retries + 1):
        attempts += 1
        step("Writing analysis code..." if attempts == 1 else f"Fixing the code (attempt {attempts})...")
        raw = client.chat(SYSTEM_PROMPT, messages, max_tokens=1500)
        code = extract_code(raw)
        step("Running the code on your data...")
        run = run_code(code, df, timeout=timeout)
        if run["ok"]:
            break
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                f"That code failed with:\n{run['error']}\n"
                "Fix it. Return only the corrected code in one python block."
            )},
        ]

    out = {"question": question, "code": code, "attempts": attempts, **run}
    if run["ok"] and explain_result:
        step("Writing the explanation...")
        out["explanation"] = explain(client, question, code, run)
    out["seconds"] = round(time.time() - started, 1)
    return out
