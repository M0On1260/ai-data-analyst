"""Best-effort sandbox for LLM-generated analysis code.

Layers:
  1. AST validation   - rejects risky imports, builtins and file/network helpers.
  2. Subprocess       - code runs in a separate `python -I` process, so a crash
                        or infinite loop can't take down the app.
  3. Timeout + limits - wall-clock timeout, and CPU/memory rlimits on Linux/macOS.
  4. Clean env        - the subprocess does NOT inherit your API key.

This is NOT a security boundary against a determined attacker. For a public
deployment, run the subprocess inside Docker (no network, read-only filesystem)
or use a hosted sandbox such as E2B.
"""
from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

RUNNER = Path(__file__).with_name("sandbox_runner.py")

ALLOWED_IMPORTS = {
    "pandas", "numpy", "plotly", "math", "statistics",
    "datetime", "re", "collections", "itertools", "calendar",
}

BLOCKED_NAMES = {
    "exec", "eval", "compile", "open", "input", "breakpoint", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
    "help", "memoryview",
}

# File / network / serialization helpers across pandas, numpy and plotly.
BLOCKED_ATTRS = {
    "to_csv", "to_pickle", "to_excel", "to_parquet", "to_json", "to_sql",
    "to_feather", "to_hdf", "to_html", "to_latex", "to_markdown", "to_clipboard",
    "to_stata", "to_xml", "to_orc",
    "write_html", "write_image", "write_json",
    "save", "savez", "savez_compressed", "load", "loadtxt", "savetxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "eval", "io",
}


class UnsafeCode(Exception):
    pass


def validate(code: str) -> None:
    """Raise SyntaxError or UnsafeCode if the code should not be run."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise UnsafeCode(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                raise UnsafeCode(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                raise UnsafeCode(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr.startswith("read_") or node.attr in BLOCKED_ATTRS:
                raise UnsafeCode(f"use of '.{node.attr}' is not allowed")


def _limits(memory_mb: int, cpu_seconds: int):
    def apply():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
        except Exception:  # not supported on every OS (e.g. macOS RLIMIT_AS)
            pass

    return apply


def run_code(code: str, df: pd.DataFrame, timeout: int = 20, memory_mb: int = 3072) -> dict:
    """Run code against df. Returns a dict with ok, and either result fields or error."""
    try:
        validate(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"SyntaxError: {e}"}
    except UnsafeCode as e:
        return {"ok": False, "error": f"Blocked by safety check: {e}"}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df.to_pickle(tmp / "df.pkl")
        (tmp / "code.py").write_text(code, encoding="utf-8")

        keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG", "LC_ALL")
        env = {k: v for k, v in os.environ.items() if k in keep}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limits(memory_mb, timeout + 10)

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(RUNNER), str(tmp)],
                capture_output=True, text=True, timeout=timeout,
                env=env, cwd=tmp, **kwargs,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Timed out after {timeout}s (code too slow or infinite loop)"}

        out_file = tmp / "out.json"
        if not out_file.exists():
            return {"ok": False, "error": (proc.stderr or "Process crashed")[-1200:]}
        data = json.loads(out_file.read_text(encoding="utf-8"))

    if not data.get("ok"):
        return {"ok": False, "error": data.get("error", "Unknown error")}

    result = {"ok": True, "kind": data["kind"], "stdout": data.get("stdout", "")}
    if data["kind"] == "table":
        result["table"] = pd.read_json(io.StringIO(data["table"]), orient="split")
        result["total_rows"] = data["total_rows"]
    else:
        result["text"] = data["text"]
    if "figure" in data:
        import plotly.io as pio

        result["figure"] = pio.from_json(data["figure"])
    return result
