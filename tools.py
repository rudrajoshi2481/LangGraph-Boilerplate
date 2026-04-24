"""
Simple tools: bash shell, PDF reader, fact checker.
Each tool returns a plain string result.
"""

import subprocess
import requests
from config import Config


def bash_tool(command: str, timeout: int = 15) -> str:
    """Run a shell command and return stdout+stderr (truncated)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout or ""
        err = result.stderr or ""
        combined = (out + ("\n[stderr]\n" + err if err else "")).strip()
        if len(combined) > 4000:
            combined = combined[:4000] + "\n... [truncated]"
        return combined or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except Exception as e:
        return f"[error] {e}"


def pdf_tool(path: str, max_chars: int = 4000) -> str:
    """Read text from a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[error] pypdf not installed. Run: pip install pypdf"
    
    try:
        reader = PdfReader(path)
        text_parts = []
        for i, page in enumerate(reader.pages):
            text_parts.append(f"--- Page {i + 1} ---\n{page.extract_text() or ''}")
        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            return "[info] PDF has no extractable text."
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return full_text
    except FileNotFoundError:
        return f"[error] file not found: {path}"
    except Exception as e:
        return f"[error] {e}"


def fact_check_tool(claim: str, ollama_client, model_name: str) -> str:
    """Ask the model to fact-check a claim. Returns 'TRUE'/'FALSE'/'UNCERTAIN' + reason."""
    prompt = (
        "You are a strict fact-checker. Evaluate the following claim and respond "
        "in this exact format:\n"
        "VERDICT: TRUE | FALSE | UNCERTAIN\n"
        "REASON: <one short sentence>\n\n"
        f"Claim: {claim}"
    )
    try:
        resp = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "think": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or "(empty response)"
    except Exception as e:
        return f"[error] {e}"
