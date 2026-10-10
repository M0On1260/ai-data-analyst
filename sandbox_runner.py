"""Runs inside the sandbox subprocess. Not meant to be imported.

Usage: python -I sandbox_runner.py <workdir>
The workdir holds df.pkl and code.py; results are written to out.json.
"""
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

workdir = Path(sys.argv[1])
out = {"ok": False}

try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    df = pd.read_pickle(workdir / "df.pkl")
    code = (workdir / "code.py").read_text(encoding="utf-8")
    scope = {"df": df, "pd": pd, "np": np, "px": px, "go": go}

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, "<analysis>", "exec"), scope)

    res = scope.get("result")
    if res is None:
        raise ValueError("The code did not set a variable named `result`.")

    if isinstance(res, pd.Series):
        res = res.to_frame(name=str(res.name) if res.name is not None else "value")
    if isinstance(res, pd.DataFrame):
        if not isinstance(res.index, pd.RangeIndex):
            res = res.reset_index()
        res.columns = [
            " ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in res.columns
        ]
        out["kind"] = "table"
        out["total_rows"] = int(len(res))
        out["table"] = res.head(500).to_json(orient="split", date_format="iso")
    else:
        out["kind"] = "value"
        if isinstance(res, np.generic):
            res = res.item()
        out["text"] = str(res)

    fig = scope.get("fig")
    if isinstance(fig, go.Figure):
        out["figure"] = pio.to_json(fig)

    out["stdout"] = printed.getvalue()[-2000:]
    out["ok"] = True
except BaseException:  # noqa: BLE001 - we want to report everything back
    out["ok"] = False
    out["error"] = traceback.format_exc()[-1200:]

(workdir / "out.json").write_text(json.dumps(out, default=str), encoding="utf-8")
