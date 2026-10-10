"""Streamlit UI. Run with: streamlit run app.py"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from analyst import analyze, build_schema
from llm import DEFAULT_CLAUDE_MODEL, DEFAULT_GEMINI_MODEL, detect_provider, env_key, get_client

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Ask questions about your data in plain English. The generated code is always shown so you can verify it.")

SAMPLE = Path("data/sales.csv")


@st.cache_data(show_spinner=False)
def load_file(name: str, content: bytes) -> pd.DataFrame:
    import io

    buf = io.BytesIO(content)
    return pd.read_excel(buf) if name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(buf)


# ---------- sidebar: key + data ----------
with st.sidebar:
    st.header("Setup")
    providers = ["Gemini (free tier)", "Claude"]
    provider = "gemini" if st.selectbox(
        "AI provider", providers, index=0 if detect_provider() == "gemini" else 1
    ) == providers[0] else "claude"
    key_var = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
    api_key = env_key(provider) or st.text_input(
        f"{'Gemini' if provider == 'gemini' else 'Anthropic'} API key",
        type="password",
        help=f"Or set the {key_var} environment variable. Get a free Gemini key at aistudio.google.com.",
    )

    model = st.text_input(
        "Model",
        value=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) if provider == "gemini"
        else os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
        help="If you get a 404 'model not found' error, Google/Anthropic retired it: paste the current model name here.",
        key=f"model_{provider}",
    )

    st.header("Data")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    use_sample = st.button("Use sample sales data")
    if use_sample:
        if not SAMPLE.exists():
            subprocess.run([sys.executable, "make_sample_data.py"], check=True)
        st.session_state["use_sample"] = True
    if uploaded:
        st.session_state["use_sample"] = False

    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.session_state["history"] = []

df, dataset_name = None, None
if uploaded:
    df, dataset_name = load_file(uploaded.name, uploaded.getvalue()), uploaded.name
elif st.session_state.get("use_sample") and SAMPLE.exists():
    df, dataset_name = pd.read_csv(SAMPLE), "sample sales data"

if df is None:
    st.info("Upload a file or click **Use sample sales data** in the sidebar to begin.")
    st.stop()

# Reset the conversation when the dataset changes.
if st.session_state.get("dataset_name") != dataset_name:
    st.session_state.update(dataset_name=dataset_name, messages=[], history=[])
st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

with st.expander(f"Data preview: {dataset_name} ({len(df):,} rows, {len(df.columns)} columns)"):
    st.dataframe(df.head(50))
    st.text("What the model sees (schema + 3 sample rows only, never the full data):")
    st.code(build_schema(df))


def render(result: dict, key: str):
    if not result.get("ok"):
        st.error(f"Couldn't produce a working answer after {result['attempts']} attempt(s).")
        st.code(result.get("error", ""), language="text")
    else:
        if result.get("explanation"):
            st.markdown(result["explanation"])
        if result.get("figure") is not None:
            st.plotly_chart(result["figure"], key=f"fig_{key}")
        if result["kind"] == "table":
            st.dataframe(result["table"])
            if result["total_rows"] > len(result["table"]):
                st.caption(f"Showing first {len(result['table'])} of {result['total_rows']:,} rows.")
        elif not result.get("explanation"):
            st.write(result["text"])
        else:
            st.caption(f"Raw result: {result['text']}")
    with st.expander("Show generated code"):
        st.code(result["code"], language="python")
        st.caption(f"Attempts: {result['attempts']}")


for i, m in enumerate(st.session_state["messages"]):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
        else:
            render(m["result"], key=str(i))

if question := st.chat_input("Ask a question about your data"):
    if not api_key:
        st.warning("Add your API key in the sidebar first.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing..."):
                result = analyze(df, question, get_client(provider, api_key, model), history=st.session_state["history"])
        except Exception as e:  # API/auth/network errors
            st.error(f"API error: {e}")
            st.stop()
        idx = len(st.session_state["messages"])
        render(result, key=str(idx))

    st.session_state["messages"].append({"role": "assistant", "result": result})
    st.session_state["history"].append((question, result["code"]))
