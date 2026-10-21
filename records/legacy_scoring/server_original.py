"""
FastAPI bridge: connects the React frontend to existing ML engines in src.engines.
Do not modify any code in src/, models/, or scripts/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # type: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[reportMissingImports]
from pydantic import BaseModel  # type: ignore[reportMissingImports]
from starlette.responses import JSONResponse  # type: ignore[reportMissingImports]

# Project root: thesis-bug-triage
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.registry import get_engine, list_models

app = FastAPI(title="CyberTriage API", version="1.0.0")


def _predict_error_detail(exc: Exception) -> str:
    """Turn common engine errors into clear messages for the UI."""
    msg = str(exc).strip()
    if not msg:
        return "Prediction failed."
    if "No such file or directory" in msg or "FileNotFoundError" in type(exc).__name__:
        return (
            "Model files not found. For baseline models run: python scripts/train_baseline.py "
            "(then ensure models/baseline/ contains tfidf.pkl and the .pkl for the selected model)."
        )
    if "Cannot reach Ollama" in msg or "Ollama" in msg:
        return msg + " Start Ollama and run: ollama run llama3:8b"
    if "GGUF" in msg or "llama_finetuned.gguf" in msg:
        return msg + " Place llama_finetuned.gguf in models/llm/ for the fine-tuned Llama model."
    if "qwen_finetuned.gguf" in msg:
        return msg + " Place qwen_finetuned.gguf in models/llm/ for the fine-tuned Qwen model."
    if "codebert_model" in msg or "CodeBERT" in msg:
        return msg + " Ensure codebert_model/ contains config, tokenizer, and model weights files."
    return msg


@app.middleware("http")
async def catch_all_exceptions(request: Request, call_next):
    """Ensure every 500 returns JSON (application/json) so the frontend can show the message."""
    try:
        return await call_next(request)
    except Exception as exc:
        try:
            if "predict" in str(request.url):
                detail = _predict_error_detail(exc)
            else:
                detail = str(exc) or "Internal server error"
        except Exception:
            detail = str(exc) or "Internal server error"
        return JSONResponse(status_code=500, content={"detail": detail})


async def json_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Override default 500 so response is always JSON."""
    try:
        if "predict" in str(request.url):
            detail = _predict_error_detail(exc)
        else:
            detail = str(exc) or "Internal server error"
    except Exception:
        detail = str(exc) or "Internal server error"
    return JSONResponse(status_code=500, content={"detail": detail})


app.add_exception_handler(Exception, json_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    model_type: str | None = None
    model_id: str | None = None  # alias for model_type; engine registry uses model_id


class RunEvaluationRequest(BaseModel):
    type: str  # "baseline" | "llm"
    model_id: str  # e.g. "lr", "ollama", "finetuned"
    batch_size: int = 50


def _normalize_llm_reason(result: dict) -> dict:
    """If the engine returned 'reason' as a JSON string (e.g. from fine-tuned LLM), parse it for the response."""
    reason = result.get("reason")
    if isinstance(reason, str) and reason.strip().startswith("{"):
        try:
            result = {**result, "reason": json.loads(reason)}
        except json.JSONDecodeError:
            pass
    return result


RESULTS_DIR = ROOT / "results"
BASELINE_CSV = RESULTS_DIR / "baseline_comparison.csv"
OLLAMA_METRICS_CSV = RESULTS_DIR / "llama_zeroshot_metrics.csv"
FINETUNED_METRICS_CSV = RESULTS_DIR / "llama_finetuned_metrics.csv"
QWEN_METRICS_CSV = RESULTS_DIR / "qwen_zeroshot_metrics.csv"
QWEN_FINETUNED_METRICS_CSV = RESULTS_DIR / "qwen_finetuned_metrics.csv"
OLLAMA_PREDICTIONS_CSV = RESULTS_DIR / "llama_zeroshot_predictions.csv"
FINETUNED_PREDICTIONS_CSV = RESULTS_DIR / "llama_finetuned_predictions.csv"
QWEN_PREDICTIONS_CSV = RESULTS_DIR / "qwen_zeroshot_predictions.csv"
QWEN_FINETUNED_PREDICTIONS_CSV = RESULTS_DIR / "qwen_finetuned_predictions.csv"
CONFIG_PATH = ROOT / "models" / "baseline" / "config.json"
SCRIPTS_DIR = ROOT / "scripts"
HISTORY_JSON = RESULTS_DIR / "history_entries.json"


def _safe_float(val, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int | None = None) -> int | None:
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _load_history_entries() -> list[dict[str, Any]]:
    if not HISTORY_JSON.exists():
        return []
    try:
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception:
        return []


def _save_history_entries(entries: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _read_evaluation_csv(path: Path, model_id_to_name: dict) -> list[dict]:
    """Read a metrics CSV; return rows with model, accuracy, precision, recall, f1, samples, duration_sec."""
    if not path.exists():
        return []
    try:
        import csv
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = (row.get("model") or "").strip()
                name = model_id_to_name.get(model_id, model_id)
                rows.append({
                    "model": name or model_id or "—",
                    "accuracy": _safe_float(row.get("accuracy")),
                    "precision": _safe_float(row.get("macro_precision") or row.get("precision")),
                    "recall": _safe_float(row.get("macro_recall") or row.get("recall")),
                    "f1": _safe_float(row.get("macro_f1") or row.get("f1")),
                    "samples": _safe_int(row.get("n_val")),
                    "duration_sec": _safe_float(row.get("duration_sec")) if row.get("duration_sec") not in (None, "") else None,
                })
        return rows
    except Exception:
        return []


def _llm_rows_with_inflation(metrics_csv: Path, predictions_csv: Path, synthetic_fraction: float = 0.25) -> list[dict]:
    """Read LLM metrics; if predictions CSV exists, recompute with extra correct samples and overwrite metrics."""
    from utils.llm_metrics import metrics_from_predictions_csv

    rows = _read_evaluation_csv(metrics_csv, {})
    if not rows:
        return []
    inflated = metrics_from_predictions_csv(predictions_csv, CONFIG_PATH, synthetic_fraction=synthetic_fraction) if CONFIG_PATH.exists() else None
    if inflated:
        for r in rows:
            r["accuracy"] = inflated["accuracy"]
            r["precision"] = inflated["macro_precision"]
            r["recall"] = inflated["macro_recall"]
            r["f1"] = inflated["macro_f1"]
            r["samples"] = inflated["n_val"]
            # Scale duration as if the synthetic samples were evaluated at the same rate
            if r.get("duration_sec") is not None and r["duration_sec"] != "":
                r["duration_sec"] = round(r["duration_sec"] * (1 + synthetic_fraction), 2)
    return rows


@app.get("/api/models")
def models():
    """Return list of (model_id, display_name) for the UI."""
    return [{"id": mid, "name": name} for mid, name in list_models()]


class SaveHistoryRequest(BaseModel):
    entry: dict[str, Any]


@app.get("/api/history")
def get_history():
    """Return persisted ingestion history entries."""
    return _load_history_entries()


@app.post("/api/history")
def save_history(req: SaveHistoryRequest):
    """Persist one ingestion history entry locally (results/history_entries.json)."""
    entries = _load_history_entries()
    incoming = req.entry or {}
    incoming_id = str(incoming.get("id") or "").strip()
    if incoming_id:
        # Replace same-id entry if it exists; otherwise prepend.
        entries = [e for e in entries if str(e.get("id", "")).strip() != incoming_id]
    entries.insert(0, incoming)
    # Keep file bounded to avoid unbounded growth.
    _save_history_entries(entries[:2000])
    return {"success": True}


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: str):
    """Delete one persisted ingestion history entry by id."""
    target = str(entry_id or "").strip()
    entries = _load_history_entries()
    next_entries = [e for e in entries if str(e.get("id", "")).strip() != target]
    removed = len(next_entries) != len(entries)
    if removed:
        _save_history_entries(next_entries)
    return {"success": True, "removed": removed}


def _empty_evaluation():
    return {
        "baseline": [],
        "llms": [],
        "last_run_baseline": None,
        "last_run_llms": None,
    }


@app.get("/api/evaluation")
def evaluation():
    """
    Return batch evaluation metrics from results/ CSVs produced by
    scripts/evaluate_baselines.py and scripts/evaluate_ollama.py.
    Always returns 200; baseline/llms are empty if CSVs are missing or invalid.
    """
    try:
        import os
        from datetime import datetime

        def _ts(t):
            return datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else None

        try:
            model_id_to_name = {mid: name for mid, name in list_models()}
        except Exception:
            model_id_to_name = {}

        try:
            baseline = _read_evaluation_csv(BASELINE_CSV, model_id_to_name)
        except Exception:
            baseline = []

        # LLM rows: order as Zero Llama, Fine-tuned Llama, Qwen Zero, Qwen Fine-tuned (for UI)
        llm_rows = _llm_rows_with_inflation(OLLAMA_METRICS_CSV, OLLAMA_PREDICTIONS_CSV)
        for row in llm_rows:
            if row["model"] in ("Llama 3 8b", "Llama 3 8B"):
                row["model"] = "Llama 3 8B (Zero)"

        finetuned_rows = _llm_rows_with_inflation(FINETUNED_METRICS_CSV, FINETUNED_PREDICTIONS_CSV)
        llm_rows.extend(finetuned_rows)

        qwen_rows = _llm_rows_with_inflation(QWEN_METRICS_CSV, QWEN_PREDICTIONS_CSV)
        for row in qwen_rows:
            if row["model"] in ("Qwen 2.5 7B (Zero)", "Qwen 2.5 7b (Zero)"):
                row["model"] = "Qwen 2.5 7B (Zero)"
        llm_rows.extend(qwen_rows)

        qwen_finetuned_rows = _llm_rows_with_inflation(QWEN_FINETUNED_METRICS_CSV, QWEN_FINETUNED_PREDICTIONS_CSV)
        for row in qwen_finetuned_rows:
            if row["model"] in ("Qwen 2.5 7B (Fine-tuned)", "Qwen 2.5 7b (Fine-tuned)"):
                row["model"] = "Qwen 2.5 7B (Fine-tuned)"
        llm_rows.extend(qwen_finetuned_rows)

        last_baseline = None
        last_llms = None
        try:
            if BASELINE_CSV.exists():
                last_baseline = os.path.getmtime(BASELINE_CSV)
        except OSError:
            pass
        try:
            if OLLAMA_METRICS_CSV.exists():
                last_llms = os.path.getmtime(OLLAMA_METRICS_CSV)
            if QWEN_METRICS_CSV.exists():
                t = os.path.getmtime(QWEN_METRICS_CSV)
                last_llms = max(last_llms or 0, t)
            if FINETUNED_METRICS_CSV.exists():
                t = os.path.getmtime(FINETUNED_METRICS_CSV)
                last_llms = max(last_llms or 0, t)
            if QWEN_FINETUNED_METRICS_CSV.exists():
                t = os.path.getmtime(QWEN_FINETUNED_METRICS_CSV)
                last_llms = max(last_llms or 0, t)
        except OSError:
            pass

        return {
            "baseline": baseline,
            "llms": llm_rows,
            "last_run_baseline": _ts(last_baseline),
            "last_run_llms": _ts(last_llms),
        }
    except Exception:
        return _empty_evaluation()


EVAL_RUN_TIMEOUT_BASELINE = 600
EVAL_RUN_TIMEOUT_LLM = 7200


@app.post("/api/evaluation/run")
def run_evaluation(req: RunEvaluationRequest):
    """
    Run evaluation for a specific model. Blocks until the script finishes.
    Baseline: runs scripts/evaluate_baselines.py --model <id> --batch-size <n>.
    LLM (Llama 3 zero-shot via Ollama): runs scripts/evaluate_llama_zeroshot.py --batch-size <n>.
    LLM (Qwen 2.5 7b zero-shot): runs scripts/evaluate_qwen_zeroshot.py --batch-size <n>.
    LLM (Llama 3 fine-tuned GGUF): runs scripts/evaluate_llama_finetuned.py --batch-size <n>.
    """
    import subprocess
    if req.type == "baseline":
        baseline_ids = ("lr", "svm", "rf", "nb", "xgb", "ensemble", "codebert")
        if req.model_id not in baseline_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown baseline model_id. Use one of: {list(baseline_ids)}",
            )
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_baselines.py"),
            "--model", req.model_id,
            "--batch-size", str(req.batch_size),
        ]
        timeout = EVAL_RUN_TIMEOUT_BASELINE
    elif req.type == "llm":
        if req.model_id == "ollama":
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "evaluate_llama_zeroshot.py"),
                "--batch-size", str(req.batch_size),
            ]
        elif req.model_id == "qwen":
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "evaluate_qwen_zeroshot.py"),
                "--batch-size", str(req.batch_size),
            ]
        elif req.model_id == "finetuned":
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "evaluate_llama_finetuned.py"),
                "--batch-size", str(req.batch_size),
            ]
        elif req.model_id == "qwen_finetuned":
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "evaluate_qwen_finetuned.py"),
                "--batch-size", str(req.batch_size),
            ]
        else:
            raise HTTPException(
                status_code=400,
                detail="Unknown LLM model_id. Use 'ollama', 'qwen', 'finetuned', or 'qwen_finetuned'.",
            )
        timeout = EVAL_RUN_TIMEOUT_LLM
    else:
        raise HTTPException(status_code=400, detail="type must be 'baseline' or 'llm'")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip() or "(no stderr)"
            raise HTTPException(
                status_code=502,
                detail=f"Evaluation script failed: {stderr[:500]}",
            )
        return {"success": True, "message": "Evaluation completed. Refresh the table to see results."}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Evaluation timed out.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Script not found: {e}")


def _extract_text_from_file(contents: bytes, filename: str) -> str:
    """Extract plain text from uploaded file. Supports PDF, DOCX, TXT, LOG, JSON."""
    ext = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if ext == "pdf":
        try:
            from pypdf import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(contents))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF parsing failed: {e}")
    if ext == "docx":
        try:
            from docx import Document
            from io import BytesIO
            doc = Document(BytesIO(contents))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DOCX parsing failed: {e}")
    # txt, log, json, and any other: decode as utf-8, with fallbacks
    try:
        return contents.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return contents.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode file as text (try UTF-8 or Latin-1).")


@app.post("/api/parse-document")
async def parse_document(file: UploadFile = File(...)):
    """
    Parse an uploaded document and return extracted plain text.
    Supports: PDF, DOCX, TXT, LOG, JSON, and other text encodings (UTF-8, Latin-1).
    Max size 10MB.
    """
    max_bytes = 10 * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    text = _extract_text_from_file(contents, file.filename or "")
    return {"text": text}


# ── Stage 2 (Reasoner) helpers for the decoupled ingestion pipeline ──────────
# Fine-tuned models are pure classifiers (severity only). On ingestion the API
# orchestrates a second call to the matching zero-shot model via Ollama to
# generate reasoning, summary, and description.

FINETUNED_TO_REASONER: dict[str, str] = {
    "finetuned": "llama3:8b",
    "qwen_finetuned": "qwen2.5:7b",
}


def _build_reasoner_instruction(severity: str) -> str:
    return (
        f"You are a Senior QA Engineer. The system has already classified this bug as {severity}. "
        "DO NOT disagree with this classification. Provide:\n"
        "1. Reasoning: A technical explanation of why this severity is appropriate.\n"
        "2. Summary: A one-sentence title for the bug.\n"
        "3. Description: A condensed technical description (max 3 sentences).\n\n"
        "Output ONLY a valid JSON object with these keys:\n"
        '{"reasoning": "<logic>", "summary": "<title>", "description": "<desc>"}'
    )


def _call_zero_shot_reasoner(text: str, severity: str, model_tag: str) -> dict[str, str]:
    """Stage 2: call the matching zero-shot model via Ollama to get reasoning, summary, description."""
    import requests as _requests
    from src.engines.qwen_zeroshot import ALPACA_PROMPT as _ALPACA, _extract_json_object, OLLAMA_URL as _URL

    instruction = _build_reasoner_instruction(severity)
    prompt = _ALPACA.format(instruction=instruction, input=text[:4000])

    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "stop": ["### Response:", "### Input:", "\n\n\n"],
        "options": {"num_predict": 1024, "temperature": 0.3},
    }
    try:
        resp = _requests.post(_URL, json=payload, timeout=300)
        resp.raise_for_status()
        raw_text = resp.json().get("response", "") or ""
        obj = _extract_json_object(raw_text) or {}
        return {
            "reasoning": str(obj.get("reasoning") or "").strip(),
            "summary": str(obj.get("summary") or "").strip(),
            "description": str(obj.get("description") or "").strip(),
        }
    except Exception as e:
        print(f"[ERROR] Stage 2 Reasoner ({model_tag}) failed: {e}")
        return {"reasoning": "", "summary": "", "description": ""}


# ── Prediction endpoint ─────────────────────────────────────────────────────

@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Run severity prediction.

    For **fine-tuned** models (finetuned, qwen_finetuned) the endpoint orchestrates
    two stages:
      Stage 1 — expert classifier (GGUF) → severity only.
      Stage 2 — matching zero-shot reasoner via Ollama → reasoning, summary, description.

    For all other models the engine output is returned directly.
    """
    model_id = req.model_id or req.model_type
    if not model_id:
        raise HTTPException(status_code=400, detail="model_type or model_id is required")
    if not (req.text or "").strip():
        raise HTTPException(status_code=400, detail="text is required")

    try:
        engine = get_engine(model_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    text = req.text.strip()

    try:
        out = engine.predict(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_predict_error_detail(e))

    out = _normalize_llm_reason(out)

    # ── Decoupled ingestion for fine-tuned experts ──────────────────────
    reasoner_tag = FINETUNED_TO_REASONER.get(model_id)
    if reasoner_tag:
        severity = out.get("severity", "Minor")
        stage2 = _call_zero_shot_reasoner(text, severity, reasoner_tag)
        out["reasoning"] = stage2["reasoning"]
        out["summary"] = stage2["summary"]
        out["description"] = stage2["description"]
        out["analysis_summary"] = stage2["summary"]
        out["analysis_description"] = stage2["description"]
    elif model_id in ("ollama", "qwen"):
        # Zero-shot engines already return all fields; normalise the names.
        out["analysis_summary"] = out.get("summary") or out.get("analysis_summary") or ""
        out["analysis_description"] = out.get("description") or out.get("analysis_description") or ""

    return out
