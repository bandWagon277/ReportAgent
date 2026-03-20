# -*- coding: utf-8 -*-
"""
Two-Agent privacy-preserving PDF report pipeline (REPORT only).

This file focuses on:
1) render_gpt_interface           – orchestrates the web request (REPORT path uses two agents)
2) process_pdf_output             – two-agent pipeline (Agent A codegen/analysis → Agent B composer/report)
3) execute_python_code            – extended to support an 'analysis' mode for Agent A runner

Additional helpers added:
- build_agentA_messages / build_agentB_messages
- extract_python_code
- summarize_csv_privacy
- persist_artifacts_and_build_manifest
- execute_composer_code (runner for Agent B)
- save_to_file (robust storage helper; noop if you already have one)
"""

import os
import re
import io
import json
import uuid
import base64
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd

from django.http import JsonResponse, HttpResponse
from django.core.cache import cache
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# ---------------------------------------------------------------------
# GLOBALS / LOGGER
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Ensure you have these in settings:
# MEDIA_ROOT / MEDIA_URL are used by default_storage

# If you already defined OUTPUT_TYPES elsewhere, keep it. Only REPORT (PDF) path is changed here.
OUTPUT_TYPES = {
    'CSV': 'csv',
    'IMAGE': 'image',
    'PDF': 'pdf',
}

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def save_to_file(filename: str, content: str | bytes, subdir: str = "tmp") -> str:
    """
    Save text/bytes into Django default_storage, return the relative storage path.
    If you already have a save_to_file, keep yours and ignore this.
    """
    folder = os.path.join(subdir, datetime.utcnow().strftime("%Y%m%d"))
    try:
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content
        path = os.path.join(folder, filename)
        saved_path = default_storage.save(path, ContentFile(content_bytes))
        return saved_path
    except Exception as e:
        logger.error(f"save_to_file error: {e}")
        raise

def extract_python_code(text: str) -> str:
    """
    Extract the best python fenced block from LLM response.
    """
    if not text:
        return ""
    t = text.strip()

    # 1) ```python ... ```
    m = re.findall(r"```(?:python|py)\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        # Pick the most code-like by length / imports
        def score(c: str) -> int:
            low = c.lower()
            keys = ['import ', 'from ', 'def ', 'class ', 'reportlab', 'matplotlib', 'bytesio']
            return sum(k in low for k in keys) + len(c) // 200
        m.sort(key=score, reverse=True)
        return m[0].strip()

    # 2) Any fenced block ```
    m2 = re.findall(r"```(.*?)```", t, flags=re.DOTALL)
    if m2:
        cands = []
        for blk in m2:
            body = blk.strip()
            low = body.lower()
            looks_py = any(k in low for k in ['import ', 'from ', 'def ', 'class ', 'reportlab', 'matplotlib'])
            looks_json = (body.startswith('{') and body.endswith('}')) or ('":' in body and '{' in body)
            looks_html = ('<' in body and '</' in body)
            if looks_py and not looks_json and not looks_html:
                cands.append(body)
        if cands:
            def score2(c: str) -> int:
                low = c.lower()
                keys = ['import ', 'from ', 'def ', 'class ', 'reportlab', 'matplotlib', 'bytesio']
                return sum(k in low for k in keys) + len(c) // 200
            cands.sort(key=score2, reverse=True)
            return cands[0].strip()

    # 3) Heuristic: collect contiguous lines starting from first import/def
    lines = t.splitlines()
    idxs = [i for i, ln in enumerate(lines) if re.search(r'^\s*(import |from |def |class )', ln)]
    if idxs:
        i0 = idxs[0]
        buf, empty = [], 0
        for ln in lines[i0:]:
            buf.append(ln)
            if ln.strip() == "":
                empty += 1
                if empty >= 3:
                    break
            else:
                empty = 0
        code = '\n'.join(buf).strip()
        if any(k in code for k in ['import ', 'from ', 'def ', 'reportlab', 'plt.']):
            return code
    return ""

def summarize_csv_privacy(csv_path: str, max_cat: int = 10) -> Dict[str, Any]:
    """
    Privacy-preserving summary: dtypes, numeric describe(), top-k categories counts.
    NO raw rows returned.
    """
    with default_storage.open(csv_path, 'r') as f:
        csv_text = f.read()
    df = pd.read_csv(io.StringIO(csv_text))

    schema = {c: str(df[c].dtype) for c in df.columns}
    numeric_desc = df.describe(include='number').round(2).to_dict()

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cats = {}
    for c in cat_cols:
        vc = df[c].astype(str).value_counts(dropna=True).head(max_cat)
        cats[c] = {str(k): int(v) for k, v in vc.items()}

    info = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "schema": schema,
        "numeric_summary": numeric_desc,
        "categorical_topk": cats,
        "missing_total": int(df.isna().sum().sum()),
    }
    return info

def build_agentA_messages(user_prompt: str, privacy_summary: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Agent A (CodeGen/Analyst) messages – produces analysis code that creates artifacts dict.
    """
    system_msg = (
        "You are a data analyst. You will NOT receive raw rows.\n"
        "Use the injected DataFrame df on the server to compute charts/tables/metrics.\n"
        "Return ONE fenced python block ONLY, that when executed sets an 'artifacts' dict with:\n"
        "  artifacts = {\n"
        "    'figures': [ {'id': str, 'title': str, 'description': str,\n"
        "                  'png_bytes': bytes OR base64(str) }, ... ],\n"
        "    'tables':  [ {'id': str, 'title': str,\n"
        "                  'csv': str OR 'rows': [[...], ...], 'headers': [..] }, ... ],\n"
        "    'metrics': { ... key: value ... },\n"
        "    'sections_implemented': [str, ...],\n"
        "    'warnings': [str, ...]\n"
        "  }\n"
        "No file I/O. Use matplotlib→BytesIO for images. ASCII text only."
    )

    user_msg = (
        f"USER REQUIREMENTS:\n{user_prompt}\n\n"
        f"DATA SUMMARY (privacy-safe):\n{json.dumps(privacy_summary, ensure_ascii=True)}\n\n"
        "IMPLEMENTATION NOTES:\n"
        "- Use the existing 'df' injected by runtime.\n"
        "- Encode figures as PNG into BytesIO. Provide raw bytes in memory OR base64 string without data URI prefix.\n"
        "- Tables must be aggregated (counts/rates), no raw rows.\n"
        "- Keep outputs small; avoid many large figures. Close figures with plt.close().\n"
        "- Set 'artifacts' as shown; NO printing/logging.\n"
        "Return only:\n```python\n# code\n```"
    )
    return [{"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}]

def build_agentB_messages(manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Agent B (Narrator/Composer) messages – produces composer code that reads manifest + loaders to build PDF.
    """
    system_msg = (
        "You are a report author. You will receive a sanitized manifest of figures/tables/metrics.\n"
        "Compose a professional PDF using ReportLab. You have:\n"
        "  - manifest: dict with lists of figures/tables/metrics (no raw data),\n"
        "  - asset_loader(fig_id) -> bytes  # returns full PNG bytes by id,\n"
        "  - table_loader(table_id) -> dict with 'headers' and 'rows' or CSV string.\n"
        "REQUIREMENTS:\n"
        "- Create a PDF (SimpleDocTemplate) and set 'pdf_data' (bytes) and 'narrative_text' (str).\n"
        "- Insert each figure with a short explanatory paragraph immediately after it (1-3 sentences).\n"
        "- Insert key tables (as ReportLab Table) with brief captions.\n"
        "- Professional formatting, ASCII text only. No internet. No file I/O.\n"
        "Return ONE fenced python block ONLY."
    )
    user_msg = (
        "MANIFEST:\n"
        + json.dumps(manifest, ensure_ascii=True) +
        "\n\nIMPLEMENTATION NOTES:\n"
        "- Use asset_loader/table_loader to fetch bytes/rows by id; do NOT assume file paths.\n"
        "- Set outputs:\n"
        "  pdf_data: bytes\n"
        "  narrative_text: str\n"
        "  used_artifacts: {'figures': [...], 'tables': [...]} (optional)\n"
        "Return only:\n```python\n# code\n```"
    )
    return [{"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}]

def persist_artifacts_and_build_manifest(artifacts: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Persist figures/tables into MEDIA storage, build a manifest (no raw data), return (manifest_id, manifest_dict).
    """
    manifest_id = str(uuid.uuid4())
    fig_dir = os.path.join("artifacts", manifest_id, "figures")
    tbl_dir = os.path.join("artifacts", manifest_id, "tables")
    os.makedirs(os.path.join(default_storage.location, fig_dir), exist_ok=True) if hasattr(default_storage, "location") else None
    os.makedirs(os.path.join(default_storage.location, tbl_dir), exist_ok=True) if hasattr(default_storage, "location") else None

    figures_manifest = []
    tables_manifest = []
    path_map = {"figures": {}, "tables": {}}

    # --- Figures ---
    for fig in artifacts.get("figures", []):
        fid = fig.get("id") or f"fig_{uuid.uuid4().hex[:8]}"
        title = fig.get("title", "")
        desc = fig.get("description", "")

        png_bytes = fig.get("png_bytes")
        if isinstance(png_bytes, str):
            # base64 string
            try:
                png_bytes = base64.b64decode(png_bytes)
            except Exception:
                png_bytes = b""

        if not isinstance(png_bytes, (bytes, bytearray)):
            logger.warning(f"Figure {fid} has no valid png_bytes; skipping")
            continue

        filename = f"{fid}.png"
        rel_path = os.path.join(fig_dir, filename)
        default_storage.save(rel_path, ContentFile(png_bytes))
        path_map["figures"][fid] = rel_path

        # Keep only metadata in manifest (no full bytes)
        preview_b64 = base64.b64encode(png_bytes[:4096]).decode('ascii')  # small head preview
        figures_manifest.append({
            "id": fid,
            "title": title,
            "caption": desc,
            "preview_b64_head": preview_b64
        })

    # --- Tables ---
    for tb in artifacts.get("tables", []):
        tid = tb.get("id") or f"tbl_{uuid.uuid4().hex[:8]}"
        title = tb.get("title", "")

        out_csv = None
        if "csv" in tb and isinstance(tb["csv"], str):
            out_csv = tb["csv"]
        elif "rows" in tb:
            rows = tb["rows"] or []
            headers = tb.get("headers") or []
            if rows and isinstance(rows, list):
                # build CSV
                buf = io.StringIO()
                if headers:
                    buf.write(",".join(map(str, headers)) + "\n")
                for r in rows:
                    buf.write(",".join(map(lambda x: str(x) if x is not None else "", r)) + "\n")
                out_csv = buf.getvalue()

        if not out_csv:
            logger.warning(f"Table {tid} has no CSV/rows; skipping")
            continue

        filename = f"{tid}.csv"
        rel_path = os.path.join(tbl_dir, filename)
        default_storage.save(rel_path, ContentFile(out_csv.encode("utf-8")))
        path_map["tables"][tid] = rel_path

        # Put small sample of rows (head) into manifest (safe)
        sample_rows = "\n".join(out_csv.splitlines()[:8])
        tables_manifest.append({
            "id": tid,
            "title": title,
            "preview_csv_head": sample_rows
        })

    manifest = {
        "manifest_id": manifest_id,
        "generated_on": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "figures": figures_manifest,
        "tables": tables_manifest,
        "metrics": artifacts.get("metrics", {}),
        "sections_implemented": artifacts.get("sections_implemented", []),
        "warnings": artifacts.get("warnings", [])
    }

    # Save manifest JSON
    manifest_path = os.path.join("manifests", f"{manifest_id}.json")
    default_storage.save(manifest_path, ContentFile(json.dumps(manifest, ensure_ascii=True).encode("utf-8")))
    cache.set(f"manifest:{manifest_id}:paths", path_map, timeout=3600)

    return manifest_id, manifest

def execute_composer_code(py_file_path: str, manifest: Dict[str, Any], manifest_paths: Dict[str, Dict[str, str]]) -> Tuple[bytes, str]:
    """
    Execute Agent B composer code with injected loaders (no CSV required).
    Returns (pdf_data bytes, narrative_text str).
    """
    # loaders
    def asset_loader(fig_id: str) -> bytes:
        path = manifest_paths["figures"].get(fig_id)
        if not path or not default_storage.exists(path):
            raise ValueError(f"Figure not found for id: {fig_id}")
        with default_storage.open(path, 'rb') as f:
            return f.read()

    def table_loader(table_id: str) -> Dict[str, Any] | str:
        path = manifest_paths["tables"].get(table_id)
        if not path or not default_storage.exists(path):
            raise ValueError(f"Table not found for id: {table_id}")
        with default_storage.open(path, 'r') as f:
            csv_text = f.read()
        # The composer can parse CSV or we can convert to headers/rows here if needed
        return csv_text

    with default_storage.open(py_file_path, 'r') as f:
        code = f.read()

    exec_globals = {
        "manifest": manifest,
        "asset_loader": asset_loader,
        "table_loader": table_loader
    }
    try:
        exec(code, exec_globals)
    except Exception as e:
        logger.error(f"Composer execution error: {e}")
        raise

    pdf_data = exec_globals.get("pdf_data")
    narrative_text = exec_globals.get("narrative_text", "")
    if not isinstance(pdf_data, (bytes, bytearray)):
        raise ValueError("Composer did not set valid pdf_data bytes.")
    return pdf_data, narrative_text

def call_openai_chat(api_key: str, messages: List[Dict[str, str]], model: str = "gpt-4o", temperature: float = 0.2, timeout: int = 120) -> str:
    """
    Minimal OpenAI Chat Completions call (no streaming). Returns content string.
    """
    import requests
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {"model": model, "messages": messages, "temperature": temperature}

    resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:500]}")
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    # Make safer (avoid accidental escapes)
    return content.replace("\\", "")

# ---------------------------------------------------------------------
# 3) RUNNER (extended)
# ---------------------------------------------------------------------

def execute_python_code(csv_file_path, py_file_path, output_type='csv',
                        csv_path=None, image_path=None,
                        dry_run=False, row_limit=None):
    """
    Extended runner:
    - When output_type == 'analysis': expect the executed code to set an 'artifacts' dict in globals.
    - For 'csv'/'image'/'pdf': keep your existing behavior via handle_execution_result.
    """
    try:
        if py_file_path is None:
            logger.error("Python file path is None.")
            raise ValueError("Python file path is None.")

        # Choose data source: prefer csv_path (real uploaded path); fallback to legacy csv_file_path
        src_csv_path = csv_path or csv_file_path
        if not src_csv_path:
            raise ValueError("CSV file path is empty.")

        with default_storage.open(src_csv_path, 'r') as csv_file:
            csv_data = csv_file.read()
        if not csv_data.strip():
            logger.error("CSV file is empty.")
            raise ValueError("CSV file is empty.")

        df = pd.read_csv(io.StringIO(csv_data))

        # dry-run sample cut
        if dry_run and row_limit and isinstance(row_limit, int) and row_limit > 0:
            df = df.head(row_limit)

        with default_storage.open(py_file_path, 'r') as py_file:
            python_code = py_file.read()

        logger.info(f"Executing Python code for {output_type} (dry_run={dry_run}, row_limit={row_limit}): {python_code[:500]}...")

        # Inject execution environment
        exec_globals = setup_execution_environment(df, output_type, csv_path=src_csv_path, image_path=image_path) \
                       if 'setup_execution_environment' in globals() else {"df": df, "csv_path": src_csv_path, "image_path": image_path}
        exec_globals['VALIDATION_MODE'] = bool(dry_run)

        try:
            exec(python_code, exec_globals)

            if dry_run:
                return {'ok': True}

            # Special branch for Agent A analysis
            if output_type == 'analysis':
                artifacts = exec_globals.get("artifacts")
                if not isinstance(artifacts, dict):
                    raise ValueError("Executed analysis did not set 'artifacts' dict.")
                return artifacts

            # Otherwise fall back to your existing output handler
            if 'handle_execution_result' in globals():
                return handle_execution_result(exec_globals, output_type)
            else:
                # Minimal fallback for pdf branch
                if output_type == 'pdf':
                    pdf_data = exec_globals.get("pdf_data")
                    if not isinstance(pdf_data, (bytes, bytearray)):
                        raise ValueError("pdf_data not produced by code.")
                    return pdf_data
                raise ValueError(f"Unknown output type without handler: {output_type}")

        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            if dry_run:
                return {'ok': False, 'error': str(e)}
            return e

    except Exception as e:
        logger.error(f"Error executing Python code on CSV data: {e}")
        if dry_run:
            return {'ok': False, 'error': str(e)}
        return e

# ---------------------------------------------------------------------
# 2) PROCESSOR (REPORT / two-agent)
# ---------------------------------------------------------------------

def process_pdf_output(generated_text: str | None,
                       csv_path: Optional[str] = None,
                       image_path: Optional[str] = None,
                       api_key: Optional[str] = None,
                       privacy_summary: Optional[Dict[str, Any]] = None,
                       user_prompt: Optional[str] = None) -> HttpResponse:
    """
    Three-agent pipeline:
      Agent 0 (Planner): user prompt + data summary → structured report plan
      Agent A: codegen (guided by plan) → analysis on real df → artifacts persisted → manifest built
      Agent B: composer (with plan + user prompt context) → render PDF from manifest only
    Returns JsonResponse with {success, result_id, output_type, preview_data, manifest_id}.
    """
    if not api_key:
        return JsonResponse({'error': 'OpenAI API key not configured'}, status=500)
    if not csv_path:
        return JsonResponse({'error': 'csv_path is required'}, status=400)

    # ----------------------
    # Privacy Summary
    # ----------------------
    try:
        if not privacy_summary:
            privacy_summary = summarize_csv_privacy(csv_path)
    except Exception as e:
        logger.error(f"Failed to build privacy summary: {e}")
        return JsonResponse({'error': f'Failed to summarize CSV: {str(e)}'}, status=500)

    effective_prompt = user_prompt or "Generate a clinical report."

    # ----------------------
    # Agent 0: Planner
    # ----------------------
    report_plan = None
    try:
        from gptapp.gpt_backend_utils import get_pdf_planner_prompt, generate_report_plan
        planner_tmpl = get_pdf_planner_prompt()
        if planner_tmpl:
            logger.info("Calling Agent 0 (Planner)...")
            report_plan = generate_report_plan(effective_prompt, privacy_summary, planner_tmpl, api_key)
            logger.info(f"Report plan: {report_plan.get('report_title', 'N/A')} "
                        f"with {len(report_plan.get('sections', []))} sections")
    except Exception as e:
        logger.warning(f"Planner failed (non-fatal, continuing without plan): {e}")

    # ----------------------
    # Agent A: CodeGen (guided by plan)
    # ----------------------
    try:
        messages_A = build_agentA_messages(effective_prompt, privacy_summary)
        logger.info("Calling Agent A (codegen)...")
        resp_A = call_openai_chat(api_key, messages_A)
        code_A = extract_python_code(resp_A)
        if not code_A:
            return JsonResponse({'error': 'Agent A did not return Python code.'}, status=400)
        codeA_path = save_to_file("agentA_code.py", code_A, subdir="agentA")
    except Exception as e:
        logger.error(f"Agent A call failed: {e}")
        return JsonResponse({'error': f'Agent A error: {str(e)}'}, status=500)

    # Run Agent A on REAL data to produce artifacts
    artifacts = execute_python_code(
        csv_file_path=None,
        py_file_path=codeA_path,
        output_type='analysis',
        csv_path=csv_path,
        image_path=image_path,
        dry_run=False
    )
    if isinstance(artifacts, Exception):
        logger.error(f"Agent A execution error: {artifacts}")
        return JsonResponse({'error': str(artifacts)}, status=500)
    if not isinstance(artifacts, dict):
        return JsonResponse({'error': 'Agent A did not produce artifacts dict.'}, status=500)

    # Persist artifacts and build manifest
    try:
        manifest_id, manifest = persist_artifacts_and_build_manifest(artifacts)
    except Exception as e:
        logger.error(f"Persist artifacts error: {e}")
        return JsonResponse({'error': f'Persist artifacts error: {str(e)}'}, status=500)

    # ----------------------
    # Agent B: Composer (with full context)
    # ----------------------
    try:
        from gptapp.gpt_backend_utils import build_enhanced_agentB_messages
        messages_B = build_enhanced_agentB_messages(
            manifest=manifest,
            instruction_template_B=None,
            user_prompt=effective_prompt,
            report_plan=report_plan,
        )
        logger.info("Calling Agent B (composer with plan context)...")
        resp_B = call_openai_chat(api_key, messages_B)
        code_B = extract_python_code(resp_B)
        if not code_B:
            return JsonResponse({'error': 'Agent B did not return Python code.'}, status=400)
        codeB_path = save_to_file("agentB_composer.py", code_B, subdir="agentB")
    except Exception as e:
        logger.error(f"Agent B call failed: {e}")
        return JsonResponse({'error': f'Agent B error: {str(e)}'}, status=500)

    # Execute composer to build final PDF
    try:
        manifest_paths = cache.get(f"manifest:{manifest_id}:paths") or {"figures": {}, "tables": {}}
        pdf_bytes, narrative_text = execute_composer_code(codeB_path, manifest, manifest_paths)
    except Exception as e:
        logger.error(f"Composer execution error: {e}")
        return JsonResponse({'error': f'Composer execution error: {str(e)}'}, status=500)

    # Cache final PDF for download
    try:
        result_id = str(uuid.uuid4())
        cache.set(f"result_{result_id}", pdf_bytes, timeout=3600)
    except Exception as e:
        logger.error(f"Caching PDF failed: {e}")
        return JsonResponse({'error': f'Caching PDF failed: {str(e)}'}, status=500)

    return JsonResponse({
        'success': True,
        'result_id': result_id,
        'output_type': 'pdf',
        'manifest_id': manifest_id,
        'preview_data': 'PDF report generated successfully',
        'narrative_excerpt': (narrative_text[:400] + '...') if narrative_text and len(narrative_text) > 400 else (narrative_text or "")
    }, status=200)

# ---------------------------------------------------------------------
# 1) VIEW (REPORT-only path changed to two agents)
# ---------------------------------------------------------------------

def render_gpt_interface(request):
    """Render the HTML interface for GPT interaction and handle form submission."""
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")

    if request.method == 'POST':
        try:
            if not request.body:
                logger.error("Empty request body")
                return JsonResponse({'error': 'Empty request body'}, status=400)

            body_preview = request.body.decode('utf-8')[:500]
            logger.info(f"Request body preview: {body_preview}")

            try:
                data = json.loads(request.body.decode('utf-8'))
                logger.info(f"Parsed JSON data keys: {list(data.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return JsonResponse({'error': f'Invalid JSON format: {str(e)}'}, status=400)

            user_prompt = (data.get('prompt') or '').strip()
            output_type = (data.get('output_type') or 'CSV').upper()
            csv_path = data.get('csv_path')
            image_path = data.get('image_path')

            logger.info(f"Parameters - prompt: {bool(user_prompt)}, output_type: {output_type}, csv_path: {csv_path}")

            # validation
            if not user_prompt:
                logger.error("No prompt provided")
                return JsonResponse({'error': 'No prompt provided'}, status=400)

            if output_type not in OUTPUT_TYPES:
                logger.error(f"Invalid output type: {output_type}")
                return JsonResponse({'error': f'Invalid output type: {output_type}. Valid types: {list(OUTPUT_TYPES.keys())}'}, status=400)

            if not csv_path:
                logger.error("csv_path is required")
                return JsonResponse({'error': 'csv_path is required'}, status=400)

            if not default_storage.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return JsonResponse({'error': f'CSV file not found: {csv_path}'}, status=400)

            api_key = os.getenv('OPENAI_API_KEY', '')
            if not api_key:
                logger.error("OpenAI API key not configured")
                return JsonResponse({'error': 'OpenAI API key not configured'}, status=500)

            # For REPORT (PDF) we run the two-agent pipeline (privacy-preserving)
            if output_type == 'PDF':
                # privacy-safe dataset summary (no raw rows)
                try:
                    privacy_summary = summarize_csv_privacy(csv_path)
                except Exception as e:
                    logger.error(f"Error generating privacy summary: {e}")
                    return JsonResponse({'error': f'Error summarizing CSV: {str(e)}'}, status=500)

                # Orchestrate two agents inside process_pdf_output (ignore generated_text param)
                result = process_pdf_output(
                    generated_text=None,
                    csv_path=csv_path,
                    image_path=image_path,
                    api_key=api_key,
                    privacy_summary=privacy_summary,
                    user_prompt=user_prompt
                )
                return result

            # For CSV/IMAGE – keep your old logic if needed, or return a friendly note:
            return JsonResponse({'error': 'This endpoint currently supports REPORT(PDF) two-agent flow only.'}, status=400)

        except Exception as e:
            logger.error(f"Unexpected error in render_gpt_interface: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return JsonResponse({
                'error': f'Internal server error: {str(e)}',
                'traceback': traceback.format_exc()
            }, status=500)

    # GET
    return render(request, 'gpt_interface.html')
