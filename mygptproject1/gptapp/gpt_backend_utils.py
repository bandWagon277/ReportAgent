# gpt_backend_utils.py
# === Consolidated imports ===
import os, re, io, json, uuid, base64, logging, tempfile
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from io import StringIO, BytesIO

# Matplotlib/Seaborn/PIL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ReportLab (PDF composition)
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Image as RLImage, PageBreak, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Django storage/cache/http
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.conf import settings

# Optional: external HTTP for OpenAI
import requests

logger = logging.getLogger(__name__)

# --- PNG helpers (signatures) ---
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# =========================
# 1) Execution & environment
# =========================
# === 修改签名：新增 csv_path, image_path，并注入到全局命名空间 ===
def setup_execution_environment(df, output_type, csv_path=None, image_path=None):
    base_globals = {
        'df': df,
        'pd': pd,
        'StringIO': StringIO,
        'default_storage': default_storage,
        'processed_data': None,
        # 新增：把真实文件路径注入，供用户代码按需读取
        'csv_path': csv_path,
        'image_path': image_path,
        'os': os,              # 方便用户代码判断/拼路径
        'BytesIO': BytesIO,    # 便于读写二进制流
        'base64': base64,      # 便于图像编码
    }

    if output_type == 'image':
        base_globals.update({
            'plt': plt,
            'sns': sns,
            'matplotlib': matplotlib,
            'Image': Image,
            'image_data': None
        })

    elif output_type == 'pdf':
        base_globals.update({
            'SimpleDocTemplate': SimpleDocTemplate,
            'Paragraph': Paragraph,
            'Spacer': Spacer,
            'RLImage': RLImage,
            'PageBreak': PageBreak,
            'Table': Table,
            'TableStyle': TableStyle,
            'getSampleStyleSheet': getSampleStyleSheet,
            'ParagraphStyle': ParagraphStyle,
            'letter': letter,
            'A4': A4,
            'inch': inch,
            'colors': colors,
            'tempfile': tempfile,
            'plt': plt,
            'sns': sns,
            'pdf_buffer': BytesIO(),
            'pdf_data': None
        })

    return base_globals

def handle_execution_result(exec_globals, output_type):
    """Handle the execution result based on output type."""
    if output_type == 'csv':
        if 'processed_data' in exec_globals and exec_globals['processed_data'] is not None:
            df = exec_globals['processed_data']
        else:
            df = exec_globals['df']
            logger.warning("Processed data not set in executed Python code. Using original DataFrame.")
        
        results_buffer = StringIO()
        df.to_csv(results_buffer, index=False)
        results_buffer.seek(0)
        return results_buffer.getvalue()
    
    elif output_type == 'image':
        if 'image_data' in exec_globals and exec_globals['image_data'] is not None:
            return exec_globals['image_data']  # Should be base64 encoded
        else:
            logger.error("Image data not generated in executed Python code.")
            raise ValueError("Image data not generated in executed Python code.")
    
    elif output_type == 'pdf':
        if 'pdf_data' in exec_globals and exec_globals['pdf_data'] is not None:
            return exec_globals['pdf_data']  # Should be PDF binary data
        else:
            logger.error("PDF data not generated in executed Python code.")
            raise ValueError("PDF data not generated in executed Python code.")
    
    else:
        raise ValueError(f"Unknown output type: {output_type}")

def execute_python_code(csv_file_path, py_file_path, output_type='csv',
                        csv_path=None, image_path=None,
                        dry_run=False, row_limit=None):
    try:
        if py_file_path is None:
            logger.error("Python file path is None.")
            raise ValueError("Python file path is None.")

        src_csv_path = csv_path or csv_file_path
        if not src_csv_path:
            raise ValueError("CSV file path is empty.")

        with default_storage.open(src_csv_path, 'r') as csv_file:
            csv_data = csv_file.read()
        if not csv_data.strip():
            logger.error("CSV file is empty.")
            raise ValueError("CSV file is empty.")

        df = pd.read_csv(StringIO(csv_data))
        if dry_run and row_limit and isinstance(row_limit, int) and row_limit > 0:
            df = df.head(row_limit)

        with default_storage.open(py_file_path, 'r') as py_file:
            python_code = py_file.read()

        logger.info(f"Executing Python code for {output_type} (dry_run={dry_run}, row_limit={row_limit}): {python_code[:500]}...")

        exec_globals = setup_execution_environment(df, output_type, csv_path=src_csv_path, image_path=image_path)
        exec_globals['VALIDATION_MODE'] = bool(dry_run)

        try:
            exec(python_code, exec_globals)

            if dry_run:
                return {'ok': True}

            # NEW: special branch for Agent A 'analysis' to fetch artifacts dict
            if output_type == 'analysis':
                artifacts = exec_globals.get("artifacts")
                if not isinstance(artifacts, dict):
                    raise ValueError("Executed analysis did not set 'artifacts' dict.")
                return artifacts

            return handle_execution_result(exec_globals, output_type)

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


# =========================
# 2) Cleaning & persistence
# =========================
import re, textwrap, unicodedata

def sanitize_python(code: str) -> str:
    """Remove markdown fences, weird unicode, tabs; dedent; normalize EOL."""
    if not code:
        return ""
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    code = re.sub(r'^\s*```(?:python)?\s*\n', '', code, flags=re.IGNORECASE)
    code = re.sub(r'\n```[\s]*$', '\n', code)
    code = ''.join(ch for ch in code if unicodedata.category(ch) != 'Cf')
    code = code.replace('\t', '    ')
    code = code.lstrip('\n')
    code = textwrap.dedent(code)
    return code.strip()


def save_to_file(file_name, content):
    """Save the given content to a file with the specified name."""
    if not content or not content.strip():
        logger.warning(f"No content to save for {file_name}")
        return None

    # === 净化步骤 ===
    content = sanitize_python(content)
    # Replace non-ASCII characters that cause encoding issues on Windows
    content = content.encode('ascii', errors='replace').decode('ascii')

    file_path = os.path.join(default_storage.location, file_name)

    with default_storage.open(file_path, 'w') as f:
        f.write(content)

    if default_storage.exists(file_path):
        logger.info(f"Saved sanitized content to {file_path}")
    else:
        logger.error(f"Failed to save content to {file_name}")

    return file_path

# =========================
# 3) Prompts & LLM I/O
# =========================
# read instruction prompt for 3 sections
def get_prompt_path(output_type):
    """Get the appropriate prompt file based on output type."""
    base_path = os.getenv('PROMPT_BASE_PATH',
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"))

    if output_type == 'CSV':
        return os.path.join(base_path, "Instruction_prompt_csv.txt")
    elif output_type == 'IMAGE':
        return os.path.join(base_path, "Instruction_prompt_image.txt")
    elif output_type == 'PDF':
        return os.path.join(base_path, "Instruction_prompt_pdf.txt")
    else:
        return os.path.join(base_path, "Instruction_prompt.txt")

def get_pdf_planner_prompt() -> str:
    """Read the Planner prompt template for Agent 0."""
    base_path = os.getenv('PROMPT_BASE_PATH',
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"))
    path_planner = os.path.join(base_path, "Instruction_prompt_pdf_Planner.txt")
    if os.path.exists(path_planner):
        with open(path_planner, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def get_pdf_dual_prompts() -> tuple[str, str]:
    """
    Read A/B templates for PDF. Fallback to existing Instruction_prompt_pdf.txt if A/B not found.
    """
    base_path = os.getenv('PROMPT_BASE_PATH',
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"))
    path_A = os.path.join(base_path, "Instruction_prompt_pdf_A.txt")
    path_B = os.path.join(base_path, "Instruction_prompt_pdf_B.txt")
    path_single = os.path.join(base_path, "Instruction_prompt_pdf.txt")

    tmpl_A = tmpl_B = ""
    try:
        if os.path.exists(path_A):
            with open(path_A, "r", encoding="utf-8") as f:
                tmpl_A = f.read().strip()
        if os.path.exists(path_B):
            with open(path_B, "r", encoding="utf-8") as f:
                tmpl_B = f.read().strip()

        # fallbacks
        if not tmpl_A or not tmpl_B:
            with open(path_single, "r", encoding="utf-8") as f:
                single = f.read().strip()
            if not tmpl_A: tmpl_A = single
            if not tmpl_B: tmpl_B = single
    except Exception as e:
        logger.warning(f"Dual prompt files not fully available, fallback to single: {e}")
        with open(path_single, "r", encoding="utf-8") as f:
            single = f.read().strip()
        tmpl_A, tmpl_B = single, single

    return tmpl_A, tmpl_B


def call_openai_chat(api_key: str, messages: list, model: str = "gpt-4o", temperature: float = 0.2, timeout: int = 120) -> str:
    import requests
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:500]}")
    return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().replace("\\", "")

def extract_json_block(text: str) -> str:
    """Extract JSON from a fenced ```json block, or try parsing the whole text."""
    if not text:
        return ""
    m = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[0].strip()
    # Try finding raw JSON object
    m2 = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m2:
        return m2.group(0).strip()
    return text.strip()


def generate_report_plan(user_prompt: str, privacy_summary: dict,
                         planner_template: str, api_key: str) -> dict:
    """
    Agent 0 (Planner): Takes a user prompt + dataset summary and produces
    a structured report plan (JSON) that guides Agent A and Agent B.
    """
    system_msg = (
        "You are a clinical research report planner. "
        "Analyze the user's request and dataset schema to produce a structured report plan. "
        "Return ONLY a JSON object in a ```json fenced block."
    )
    user_content = f"""{planner_template}

### User Request
{user_prompt}

### Dataset Summary (Privacy-Preserving)
{json.dumps(privacy_summary, ensure_ascii=True, default=str)}
"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    raw = call_openai_chat(api_key, messages, timeout=90)
    json_str = extract_json_block(raw)
    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Planner JSON parse failed: {e}. Using raw text as fallback.")
        plan = {"raw_plan": json_str, "parse_error": str(e)}
    return plan


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # 1) ```python ...```
    m = re.findall(r"```(?:python|py)\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        def score(c: str) -> int:
            low = c.lower()
            keys = ['import ', 'from ', 'def ', 'class ', 'reportlab', 'matplotlib', 'bytesio']
            return sum(k in low for k in keys) + len(c) // 200
        m.sort(key=score, reverse=True)
        return m[0].strip()

    # 2) Any fenced ```
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

    # 3) Heuristic
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
        code = "\n".join(buf).strip()
        if any(k in code for k in ['import ', 'from ', 'def ', 'reportlab', 'plt.']):
            return code
    return ""

def build_agentA_messages(user_prompt: str, privacy_summary: dict, instruction_template: str | None = None):
    sys = (
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
        "No file I/O. Use matplotlib→BytesIO for images. ASCII text only.\n"
        "\n"
        "# JSON-SERIALIZABLE CONSTRAINTS (MANDATORY)\n"
        "- All values in artifacts MUST be JSON-serializable by Python's json module.\n"
        "- Use ONLY Python built-ins: str, int, float, bool, None, list, dict.\n"
        "- Convert numpy/pandas scalars via int(...), float(...), bool(...), or .item().\n"
        "- Replace NaN/NaT with None.\n"
        "- Do NOT put Timestamp/Index/Series/DataFrame inside artifacts; aggregate to numbers or small lists.\n"
    )

    tmpl = f"\n\nINSTRUCTION TEMPLATE:\n{instruction_template}\n" if instruction_template else ""
    user = (
        f"USER REQUIREMENTS:\n{user_prompt}\n"
        f"{tmpl}\n"
        f"DATA SUMMARY (privacy-safe):\n{json.dumps(privacy_summary, ensure_ascii=True)}\n\n"
        "IMPLEMENTATION NOTES:\n"
        "- Use the existing 'df' injected by runtime (full dataset in production run).\n"
        "- Encode figures as PNG into BytesIO; provide raw bytes or base64 string (no data URI prefix).\n"
        "- Tables must be aggregated (counts/rates), no raw rows.\n"
        "- Close figures with plt.close(). Keep outputs small.\n"
        "- Set 'artifacts' as specified; no prints/logs.\n"
        "Return only:\n```python\n# code\n```"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def build_agentB_messages(manifest: dict, instruction_template: str | None = None):
    sys = (
        "You are a report author. You will receive a sanitized manifest of figures/tables/metrics.\n"
        "Compose a professional PDF using ReportLab. You have:\n"
        "  - manifest: dict with lists of figures/tables/metrics (no raw data),\n"
        "  - asset_loader(fig_id) -> bytes  # returns full PNG bytes by id,\n"
        "  - table_loader(table_id) -> CSV text or {'headers': [...], 'rows': [...]}.\n"
        "REQUIREMENTS:\n"
        "- Build a PDF (SimpleDocTemplate) and set 'pdf_data' (bytes) and 'narrative_text' (str).\n"
        "- Insert each figure with a short explanatory paragraph immediately after it (1-3 sentences).\n"
        "- Insert key tables with brief captions using ReportLab Table.\n"
        "- Professional formatting, ASCII text only. No internet. No file I/O.\n"
        "Return ONE fenced python block ONLY.\n"
        "Image bytes must be retrieved via asset_loader(fig_id) .\n"
        "Using the preview_b64_head in the manifest to render images is strictly prohibited; it is a truncated preview.\n"
        "A clear error (with fig_id) should be thrown when a non-PNG/truncated image is encountered."
    )
    user = (
        (f"COMPOSITION TEMPLATE:\n{instruction_template}\n\n" if instruction_template else "") +
        "MANIFEST:\n" + json.dumps(manifest, ensure_ascii=True) +
        "\n\nIMPLEMENTATION NOTES:\n"
        "- Use asset_loader/table_loader to fetch bytes/rows by id; do NOT assume file paths.\n"
        "- Set outputs:\n"
        "  pdf_data: bytes\n"
        "  narrative_text: str\n"
        "  used_artifacts: {'figures': [...], 'tables': [...]} (optional)\n"
        "Return only:\n```python\n# code\n```"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def build_enhanced_agentB_messages(manifest: Dict[str, Any],
                                 instruction_template_B: str | None,
                                 user_prompt: str | None = None,
                                 report_plan: dict | None = None) -> List[Dict[str, str]]:
    """
    Build enhanced Agent B messages with statistical data from manifest,
    plus the original user prompt and report plan for full context.
    """
    import json

    # Use instruction template as system prompt, with fallback
    enhanced_system_prompt = instruction_template_B or """
    You are a report author. You receive a manifest with detailed statistical summaries for each figure and table.
    Use the data_summary field to write precise, data-driven insights.
    Always cite specific numbers from the data_summary. Never write generic statements.
    """

    # Extract statistical summaries for the prompt
    statistical_summaries = {
        "figures_data": [],
        "tables_data": []
    }

    if "figures" in manifest:
        for fig in manifest["figures"]:
            if "data_summary" in fig:
                logger.info(f"Agent B context - figure data_summary for {fig['id']}: {list(fig['data_summary'].keys())}")
                statistical_summaries["figures_data"].append({
                    "id": fig["id"],
                    "title": fig["title"],
                    "description": fig.get("description", ""),
                    "data_summary": fig["data_summary"]
                })

    if "tables" in manifest:
        for tbl in manifest["tables"]:
            if "data_summary" in tbl:
                statistical_summaries["tables_data"].append({
                    "id": tbl["id"],
                    "title": tbl["title"],
                    "headers": tbl.get("headers", []),
                    "data_summary": tbl["data_summary"]
                })

    # Build context sections
    context_parts = []

    # 1. Original user intent
    if user_prompt:
        context_parts.append(f"ORIGINAL USER REQUEST:\n{user_prompt}")

    # 2. Report plan from Planner agent
    if report_plan and isinstance(report_plan, dict):
        plan_summary = {
            "report_title": report_plan.get("report_title", ""),
            "study_context": report_plan.get("study_context", ""),
            "cohort_description": report_plan.get("cohort_description", ""),
            "clinical_context": report_plan.get("clinical_context", ""),
            "sections": [
                {"title": s.get("title", ""), "description": s.get("description", "")}
                for s in report_plan.get("sections", [])
            ],
            "statistical_methods": report_plan.get("statistical_methods", []),
            "key_comparisons": report_plan.get("key_comparisons", []),
        }
        context_parts.append(
            f"REPORT PLAN (use this to structure the PDF and write the introduction/discussion):\n"
            f"{json.dumps(plan_summary, indent=2)}"
        )

    # 3. Manifest overview
    context_parts.append(
        f"MANIFEST OVERVIEW:\n"
        f"{json.dumps({k: v for k, v in manifest.items() if k not in ['figures', 'tables']}, indent=2)}"
    )

    # 4. Figures and tables with statistical data
    context_parts.append(
        f"FIGURES WITH STATISTICAL DATA:\n{json.dumps(statistical_summaries['figures_data'], indent=2)}"
    )
    context_parts.append(
        f"TABLES WITH STATISTICAL DATA:\n{json.dumps(statistical_summaries['tables_data'], indent=2)}"
    )

    # 5. Instructions
    context_parts.append(
        "COMPOSITION INSTRUCTIONS:\n"
        "- Structure the PDF following the report plan sections (if provided)\n"
        "- Use the original user request to understand intent and emphasis\n"
        "- Use data_summary for each figure/table to write specific, data-driven insights\n"
        "- Include exact numbers, percentages, and statistics in your analysis\n"
        "- Write 3-4 sentences per figure/table with clinical implications\n"
        "- Place the analytical paragraph immediately after each figure/table\n"
        "- Write the Introduction referencing study_context and cohort_description from the plan\n"
        "- Write the Discussion connecting findings to clinical_context from the plan\n"
        "- Use the statistical_methods list from the plan to describe methodology\n"
        "\n"
        "Generate the PDF composer code that uses asset_loader and table_loader functions "
        "(already provided in environment)."
    )

    user_msg = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": user_msg}
    ]

    return messages

def generate_agent_a_code(user_prompt, privacy_summary, instruction_template_A, api_key,
                          report_plan=None):
    """Generate Agent A analysis code, optionally guided by a report plan from Agent 0."""
    system_msg = "You are a senior data analyst and Python expert."

    plan_section = ""
    if report_plan and isinstance(report_plan, dict):
        plan_section = (
            "\n### Report Plan (from Planner Agent)\n"
            "Follow this plan to decide which sections, figures, and tables to produce. "
            "Use the exact column names, stratification variables, and statistical tests specified.\n"
            f"{json.dumps(report_plan, indent=2, ensure_ascii=True)}\n"
        )

    user_content = f"""
{instruction_template_A}

### User Requirements
{user_prompt}
{plan_section}
### Data Context (Privacy-Preserving Summary)
{privacy_summary}

Generate Python code that analyzes the data according to the user requirements and report plan.
"""
    
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        "model": "gpt-4o", 
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2
    }
    
    resp = requests.post('https://api.openai.com/v1/chat/completions',
                        headers=headers, json=payload, timeout=120)
    
    if resp.status_code != 200:
        raise Exception(f"OpenAI API error: {resp.status_code} - {resp.text}")
    
    content = resp.json()
    generated_code = content.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
    
    return generated_code


# =========================
# 4) CSV summaries
# =========================
# simplify data summary generation
def summarize_csv_for_prompt(csv_path, max_rows=20, max_cols=60):
    """生成更适合报告生成的 CSV 数据摘要"""
    try:
        with default_storage.open(csv_path, 'r') as f:
            csv_text = f.read()
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise

    try:
        df = pd.read_csv(StringIO(csv_text))
        logger.info(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        raise

    # 如果列太多，选择前N列
    if df.shape[1] > max_cols:
        df_display = df.iloc[:, :max_cols]
        col_note = f" (showing first {max_cols} of {df.shape[1]} columns)"
    else:
        df_display = df
        col_note = ""

    # 数据类型简化
    dtypes = df.dtypes.astype(str).to_dict()
    def simplify_dtype(dtype_str):
        dtype_lower = dtype_str.lower()
        if 'float' in dtype_lower: return 'numeric'
        if 'int' in dtype_lower: return 'numeric'
        if 'bool' in dtype_lower: return 'boolean'
        if 'date' in dtype_lower or 'time' in dtype_lower: return 'datetime'
        return 'text'
    
    simplified_dtypes = {col: simplify_dtype(dtype) for col, dtype in dtypes.items()}

    # 基础统计
    try:
        numeric_summary = df.describe(include='number').round(2).to_string()
    except Exception:
        numeric_summary = "No numeric columns for statistical summary."

    try:
        categorical_summary = df.select_dtypes(include=['object', 'category']).describe().to_string()
    except Exception:
        categorical_summary = "No categorical columns for summary."

    # 构建摘要
    summary = f"""## DATASET OVERVIEW
    Total Records: {df.shape[0]:,}
    Total Columns: {df.shape[1]}{col_note}

    ## COLUMN INFORMATION
    {', '.join([f"{col} ({dtype})" for col, dtype in simplified_dtypes.items()])}

    ## SAMPLE DATA (First {min(max_rows, len(df))} rows)
    {df_display.head(max_rows).to_csv(index=False)}

    ## NUMERIC COLUMNS STATISTICS
    {numeric_summary}

    ## CATEGORICAL COLUMNS SUMMARY  
    {categorical_summary}

    ## DATA QUALITY NOTES
    Missing Values: {df.isnull().sum().sum()} total missing values
    Duplicate Rows: {df.duplicated().sum()} duplicate rows
    """
    
    return summary

def summarize_csv_privacy(csv_path: str, max_cat: int = 10) -> dict:
    """
    Build a privacy-preserving summary: schema, numeric describe, top-k categories.
    Does NOT return raw sample rows.
    """
    with default_storage.open(csv_path, 'r') as f:
        csv_text = f.read()
    df = pd.read_csv(StringIO(csv_text))

    schema = {c: str(df[c].dtype) for c in df.columns}
    numeric_desc = {}
    try:
        numeric_desc = df.describe(include='number').round(2).to_dict()
    except Exception:
        numeric_desc = {}

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cats = {}
    for c in cat_cols:
        vc = df[c].astype(str).value_counts(dropna=True).head(max_cat)
        cats[c] = {str(k): int(v) for k, v in vc.items()}

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "schema": schema,
        "numeric_summary": numeric_desc,
        "categorical_topk": cats,
        "missing_total": int(df.isna().sum().sum()),
    }

# =========================
# 5) Model output parsing
# =========================
def process_prompt(generated_text: str):
    """
    Extract Python code for CSV processing from LLM output.
    We no longer consume inline CSV; we return empty csv_data/sas_code for compatibility.

    Return: (csv_data: str, sas_code: str, python_code: str)
    """
    if not generated_text:
        return '', '', ''

    text = generated_text.strip()
    csv_data = ''   # deprecated: we now use real uploaded df
    sas_code = ''   # rarely used; keep for backward-compat
    python_code = ''

    # ---- 0) Try to extract SAS fenced block (optional/back-compat) ----
    try:
        m_sas = re.findall(r"```(?:sas)\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m_sas:
            # 选第一段即可；不强求完整
            sas_code = m_sas[0].strip()
    except Exception:
        pass

    # ---- Scoring：更像 CSV/IMAGE/报告生成的 Python 得更高分 ----
    def score_py(code: str) -> int:
        low = code.lower()
        keys = [
            'import ', 'from ', 'def ', 'class ',
            'pandas', 'pd.read_csv', 'df.', 'to_csv', 'processed_data',
            # image/report libs appear sometimes; they still indicate python
            'matplotlib', 'plt.', 'seaborn',
            'io.bytesio', 'bytesio',
        ]
        return sum(k in low for k in keys) + len(code) // 200

    # ---- 1) ```python ...``` / ```py ...``` ----
    blocks_lang = re.findall(r"```(?:python|py)\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks_lang:
        blocks_lang.sort(key=score_py, reverse=True)
        python_code = blocks_lang[0].strip()
        logger.info("process_prompt: selected python fenced block by lang tag")
        return csv_data, sas_code, python_code

    # ---- 2) 任意 ```...```：挑“像 Python”的块 ----
    any_blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    if any_blocks:
        candidates = []
        for blk in any_blocks:
            body = blk.strip()
            low = body.lower()

            looks_python = any([
                'import ' in low, 'from ' in low, 'def ' in low, 'class ' in low,
                'pandas' in low, 'df.' in low, 'to_csv' in low, 'processed_data' in low,
                'matplotlib' in low, 'plt.' in low, 'seaborn' in low
            ])
            looks_json = (body.startswith('{') and body.endswith('}')) or ('":' in body and '{' in body)
            looks_html = ('<' in body and '</' in body)

            if looks_python and not looks_json and not looks_html:
                candidates.append(body)

        if candidates:
            candidates.sort(key=score_py, reverse=True)
            python_code = candidates[0].strip()
            logger.info("process_prompt: selected python fenced block by heuristic")
            return csv_data, sas_code, python_code

    # ---- 3) 启发式：围绕 import/def 收集连续代码 ----
    lines = text.splitlines()
    py_idx = [i for i, ln in enumerate(lines) if re.search(r'^\s*(import |from |def |class )', ln)]
    if py_idx:
        i0 = py_idx[0]
        buf, empty_streak = [], 0
        for ln in lines[i0:]:
            buf.append(ln)
            if ln.strip() == '':
                empty_streak += 1
                if empty_streak >= 3:
                    break
            else:
                empty_streak = 0
        code = '\n'.join(buf).strip()
        if any(s in code for s in ['import ', 'from ', 'def ', 'pandas', 'df.', 'to_csv', 'processed_data']):
            python_code = code
            logger.info("process_prompt: selected python code by line sweep")
            return csv_data, sas_code, python_code

    # ---- 如果还没抓到，就返回空 ----
    logger.warning("process_prompt: no python code detected")
    return csv_data, sas_code, python_code

def process_image_prompt(generated_text: str):
    """
    Extract Python code for IMAGE generation from LLM output.
    We ignore inline CSV. Prefer blocks that produce `image_data` (base64 string).

    Return: (csv_data: str, python_code: str)
    """
    if not generated_text:
        return '', ''

    text = generated_text.strip()
    csv_data = ''      # deprecated: we now use real uploaded df
    python_code = ''

    def score_img_py(code: str) -> int:
        low = code.lower()
        keys = [
            'import ', 'from ', 'def ', 'class ',
            'matplotlib', 'plt.', 'seaborn', 'sns.',
            'bytesio', 'io.bytesio', 'base64', 'image_data',  # 产物变量
            'df.', 'pandas'
        ]
        # 强化要求 image_data
        bonus = 3 if 'image_data' in low else 0
        return sum(k in low for k in keys) + bonus + len(code) // 200

    # ---- 1) ```python ...``` ----
    blocks_lang = re.findall(r"```(?:python|py)\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks_lang:
        blocks_lang.sort(key=score_img_py, reverse=True)
        python_code = blocks_lang[0].strip()
        logger.info("process_image_prompt: selected python fenced block by lang tag")
        return csv_data, python_code

    # ---- 2) 任意 ```...```：挑“像绘图+image_data”的 Python 块 ----
    any_blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    if any_blocks:
        candidates = []
        for blk in any_blocks:
            body = blk.strip()
            low = body.lower()

            looks_python = any([
                'import ' in low, 'from ' in low, 'def ' in low, 'class ' in low,
                'matplotlib' in low, 'plt.' in low, 'seaborn' in low, 'sns.' in low,
                'bytesio' in low, 'base64' in low, 'image_data' in low, 'pandas' in low, 'df.' in low
            ])
            looks_json = (body.startswith('{') and body.endswith('}')) or ('":' in body and '{' in body)
            looks_html = ('<' in body and '</' in body)

            if looks_python and not looks_json and not looks_html:
                candidates.append(body)

        if candidates:
            candidates.sort(key=score_img_py, reverse=True)
            python_code = candidates[0].strip()
            logger.info("process_image_prompt: selected python fenced block by heuristic")
            return csv_data, python_code

    # ---- 3) 启发式：围绕 import/def 收集连续代码 ----
    lines = text.splitlines()
    py_idx = [i for i, ln in enumerate(lines) if re.search(r'^\s*(import |from |def |class )', ln)]
    if py_idx:
        i0 = py_idx[0]
        buf, empty_streak = [], 0
        for ln in lines[i0:]:
            buf.append(ln)
            if ln.strip() == '':
                empty_streak += 1
                if empty_streak >= 3:
                    break
            else:
                empty_streak = 0
        code = '\n'.join(buf).strip()
        if any(s in code.lower() for s in ['matplotlib', 'plt.', 'seaborn', 'bytesio', 'base64', 'image_data']):
            python_code = code
            logger.info("process_image_prompt: selected python code by line sweep")
            return csv_data, python_code

    logger.warning("process_image_prompt: no python code detected")
    return csv_data, python_code

def process_pdf_prompt(generated_text: str) -> str:
    """
    Robustly extract a single Python code block from LLM response.

    Strategy (in order):
      1) Fenced blocks with python/lang tag: ```python ...``` / ```py ...```
      2) Any fenced block ```...``` that looks like python/reportlab/matplotlib code
      3) Heuristic: contiguous lines around 'import ' / 'from ' / 'def ' / 'class ' etc.

    Returns: code string (no fences, no language tag), or '' if not found.
    """
    if not generated_text:
        return ''

    text = generated_text.strip()

    # --- 1) ```python ...``` / ```py ...``` 语言标记的代码块 ---
    pattern_lang = re.compile(
        r"```(?:python|py)\s*(.*?)```",
        flags=re.DOTALL | re.IGNORECASE
    )
    blocks = pattern_lang.findall(text)
    # 如果有多个，优先选“更像 Python/报告生成”的那个
    def score_py(code: str) -> int:
        low = code.lower()
        keys = [
            'import ', 'from ', 'def ', 'class ',
            'reportlab', 'simpledoctemplate', 'platypus', 'paragraph', 'rlimage',
            'matplotlib', 'plt.', 'bytesio', 'pdf_data', 'narrative_text'
        ]
        return sum(k in low for k in keys) + len(code) // 200  # 代码长一点也略加分

    if blocks:
        blocks_sorted = sorted(blocks, key=score_py, reverse=True)
        code = blocks_sorted[0].strip()
        return code

    # --- 2) 任意 ```...``` 的代码块，挑看起来像 Python 的 ---
    pattern_any = re.compile(r"```(.*?)```", flags=re.DOTALL)
    any_blocks = pattern_any.findall(text)
    if any_blocks:
        # 过滤掉明显不是代码/或是 JSON/HTML 的
        candidates = []
        for blk in any_blocks:
            body = blk.strip()
            low = body.lower()
            # 看起来像 Python 的启发式条件
            looks_python = any([
                'import ' in low, 'from ' in low, 'def ' in low, 'class ' in low,
                'reportlab' in low, 'simpledoctemplate' in low, 'plt.' in low
            ])
            looks_json = (body.startswith('{') and body.endswith('}')) or ('":' in body and '{' in body)
            looks_html = ('<' in body and '>' in body and '</' in body)

            if looks_python and not looks_json and not looks_html:
                candidates.append(body)

        if candidates:
            candidates_sorted = sorted(candidates, key=score_py, reverse=True)
            return candidates_sorted[0].strip()

    # --- 3) 启发式：围绕关键行抽取“连续代码段” ---
    lines = text.splitlines()
    py_lines_idx = [i for i, ln in enumerate(lines) if re.search(r'^\s*(import |from |def |class )', ln)]
    if py_lines_idx:
        i0 = max(0, py_lines_idx[0] - 0)
        # 往后收集到空行过多时停止
        buf, empty_streak = [], 0
        for ln in lines[i0:]:
            buf.append(ln)
            if ln.strip() == '':
                empty_streak += 1
                if empty_streak >= 3:
                    break
            else:
                empty_streak = 0
        code = '\n'.join(buf).strip()
        # 简单校验像不像代码
        if len(code) > 0 and ('import ' in code or 'def ' in code or 'reportlab' in code or 'plt.' in code):
            return code

    return ''

# =========================
# 6) Artifacts + Composer
# =========================
def to_json_safe(obj):
    """Recursively convert numpy/pandas scalars to built-in JSON-serializable Python types."""
    # numpy / pandas scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val):
            return None
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        # ISO string
        return obj.isoformat()
    # None/NaN
    try:
        # pd.isna works for many scalars; guard against raising on non-supported
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # containers
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]

    # everything else (str, int, float, bool, etc.)
    return obj

import base64

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

def is_png_bytes(b: bytes) -> bool:
    return isinstance(b, (bytes, bytearray)) and len(b) > 8 and b.startswith(PNG_SIGNATURE)

def try_decode_base64(s: str) -> bytes:
    # strip data URI if present
    s = s.strip()
    if s.lower().startswith("data:image/"):
        # e.g. data:image/png;base64,XXXX
        parts = s.split(",", 1)
        s = parts[1] if len(parts) == 2 else s
    # strict decode first; fallback to non-strict for leniency
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return base64.b64decode(s)

def ensure_png_bytes(maybe_bytes_or_b64) -> bytes:
    """
    Accept bytes/bytearray or base64 string, return raw PNG bytes.
    Raise ValueError if cannot obtain valid PNG bytes.
    """
    if isinstance(maybe_bytes_or_b64, (bytes, bytearray)):
        b = bytes(maybe_bytes_or_b64)
        if not is_png_bytes(b):
            raise ValueError("Provided bytes are not valid PNG (bad signature).")
        return b
    if isinstance(maybe_bytes_or_b64, str):
        b = try_decode_base64(maybe_bytes_or_b64)
        if not is_png_bytes(b):
            raise ValueError("Decoded base64 is not valid PNG (bad signature).")
        return b
    raise ValueError("png_bytes must be bytes or base64 string")


# --- NEW: persist artifacts and build a manifest + path map (no raw bytes in manifest) ---
def persist_artifacts_and_build_manifest(artifacts: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced version that preserves data_summary in manifest while storing images separately
    """
    import uuid
    from django.core.files.base import ContentFile
    from django.core.cache import cache
    
    manifest_id = str(uuid.uuid4())
    manifest_paths = {"figures": {}, "tables": {}}
    
    # Enhanced manifest with data_summary preserved
    manifest = {
        "id": manifest_id,
        "figures": [],
        "tables": artifacts.get("tables", []),  # Tables already have data_summary
        "metrics": artifacts.get("metrics", {}),
        "sections_implemented": artifacts.get("sections_implemented", []),
        "warnings": artifacts.get("warnings", [])
    }
    
    # Process figures: store PNG, keep data_summary in manifest
    if "figures" in artifacts:
        for fig in artifacts["figures"]:
            try:
                # Store PNG bytes separately
                png_bytes = fig.get("png_bytes")
                if png_bytes:
                    if isinstance(png_bytes, str):  # base64 string
                        import base64
                        png_bytes = base64.b64decode(png_bytes)
                    
                    # Save to storage
                    fig_filename = f"figures/{manifest_id}_{fig['id']}.png"
                    fig_file = ContentFile(png_bytes, name=fig_filename)
                    fig_path = default_storage.save(fig_filename, fig_file)
                    manifest_paths["figures"][fig["id"]] = fig_path
                
                # Add to manifest WITHOUT png_bytes but WITH data_summary
                manifest_fig = {
                    "id": fig["id"],
                    "title": fig.get("title", ""),
                    "description": fig.get("description", ""),
                    "data_summary": fig.get("data_summary", {})  # Keep statistical data
                }
                manifest["figures"].append(manifest_fig)
                
            except Exception as e:
                logger.warning(f"Failed to process figure {fig.get('id', 'unknown')}: {e}")
    
    # Process tables: save CSV files, keep data_summary in manifest  
    if "tables" in artifacts:
        for tbl in artifacts["tables"]:
            try:
                # Save CSV content
                if "headers" in tbl and "rows" in tbl:
                    import csv
                    from io import StringIO
                    csv_content = StringIO()
                    writer = csv.writer(csv_content)
                    writer.writerow(tbl["headers"])
                    writer.writerows(tbl["rows"])
                    
                    tbl_filename = f"tables/{manifest_id}_{tbl['id']}.csv"
                    tbl_file = ContentFile(csv_content.getvalue().encode('utf-8'), name=tbl_filename)
                    tbl_path = default_storage.save(tbl_filename, tbl_file)
                    manifest_paths["tables"][tbl["id"]] = tbl_path
                    
            except Exception as e:
                logger.warning(f"Failed to process table {tbl.get('id', 'unknown')}: {e}")
    
    # Cache paths for composer
    cache.set(f"manifest:{manifest_id}:paths", manifest_paths, timeout=3600)
    cache.set(f"manifest:{manifest_id}", manifest, timeout=3600)
    
    return manifest_id, manifest



# --- NEW: execute Agent B composer code (no CSV injection needed) ---
def execute_composer_code(py_file_path: str, manifest: Dict[str, Any], manifest_paths: Dict[str, Dict[str, str]]) -> Tuple[bytes, str]:
    def asset_loader(fig_id: str) -> bytes:
        path = manifest_paths["figures"].get(fig_id)
        if not path or not default_storage.exists(path):
            raise ValueError(f"Figure not found for id: {fig_id}")
        with default_storage.open(path, 'rb') as f:
            b = f.read()
        if not is_png_bytes(b):
            raise ValueError(f"Figure {fig_id} bytes are not valid PNG")
        return b

    def table_loader(table_id: str):
        """Load table data and return as list-of-lists (ready for ReportLab Table)."""
        # First try manifest_paths (file-based)
        path = manifest_paths["tables"].get(table_id)
        if path and default_storage.exists(path):
            import csv as _csv
            with default_storage.open(path, 'r') as f:
                reader = _csv.reader(f)
                return [row for row in reader]
        # Fallback: look in manifest tables for headers/rows
        for tbl in manifest.get("tables", []):
            if tbl.get("id") == table_id:
                if "headers" in tbl and "rows" in tbl:
                    return [tbl["headers"]] + tbl["rows"]
        raise ValueError(f"Table not found for id: {table_id}")

    with default_storage.open(py_file_path, 'r') as f:
        code = f.read()

    # Pre-inject common reportlab imports so LLM-generated code doesn't fail
    # on missing imports (a frequent issue with generated composer code).
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Image as RLImage, PageBreak, Table, TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib import colors

    exec_globals = {
        "manifest": manifest,
        "asset_loader": asset_loader,
        "table_loader": table_loader,
        # pre-injected reportlab symbols
        "letter": letter, "A4": A4, "inch": inch,
        "SimpleDocTemplate": SimpleDocTemplate, "Paragraph": Paragraph,
        "Spacer": Spacer, "RLImage": RLImage, "PageBreak": PageBreak,
        "Table": Table, "TableStyle": TableStyle,
        "getSampleStyleSheet": getSampleStyleSheet, "ParagraphStyle": ParagraphStyle,
        "TA_CENTER": TA_CENTER, "TA_JUSTIFY": TA_JUSTIFY,
        "TA_LEFT": TA_LEFT, "TA_RIGHT": TA_RIGHT,
        "colors": colors,
    }
    try:
        exec(code, exec_globals)
    except (ValueError, Exception) as e:
        err_msg = str(e)
        # Common LLM mistake: multiple <bullet> tags in one Paragraph.
        # Auto-fix by replacing <bullet>/</ bullet> with plain "- " dashes and retry.
        if "bullet" in err_msg:
            logger.warning("Auto-fixing <bullet> tag errors in composer code...")
            import re as _re
            fixed_code = _re.sub(r'<bullet>\s*', '- ', code)
            fixed_code = _re.sub(r'</bullet>', '', fixed_code)
            fixed_code = fixed_code.replace('styles[\'Bullet\']', 'styles[\'Normal\']')
            fixed_code = fixed_code.replace('styles["Bullet"]', 'styles["Normal"]')
            exec_globals_retry = dict(exec_globals)
            try:
                exec(fixed_code, exec_globals_retry)
                exec_globals = exec_globals_retry
            except Exception as retry_err:
                logger.error(f"Composer retry also failed: {retry_err}")
                raise
        else:
            logger.error(f"Composer execution error: {e}")
            raise

    pdf_data = exec_globals.get("pdf_data")
    narrative_text = exec_globals.get("narrative_text", "")
    if not isinstance(pdf_data, (bytes, bytearray)):
        raise ValueError("Composer did not set valid pdf_data bytes.")
    return pdf_data, narrative_text

from zipfile import ZipFile, ZIP_DEFLATED

def build_artifacts_zip(manifest_id: str) -> bytes:
    paths = cache.get(f"manifest:{manifest_id}:paths") or {"figures": {}, "tables": {}}
    buf = io.BytesIO()
    with ZipFile(buf, 'w', ZIP_DEFLATED) as z:
        # add figures
        for fid, rel in paths["figures"].items():
            if default_storage.exists(rel):
                with default_storage.open(rel, 'rb') as f:
                    z.writestr(f"figures/{fid}.png", f.read())
        # add tables
        for tid, rel in paths["tables"].items():
            if default_storage.exists(rel):
                with default_storage.open(rel, 'r') as f:
                    z.writestr(f"tables/{tid}.csv", f.read())
    buf.seek(0)
    return buf.read()

# =========================
# 7) Output helpers (optional service-layer)
# =========================
def get_script_path(output_type):
    """Get the script path based on output type."""
    if output_type == 'csv':
        return os.path.join(default_storage.location, "code.py")
    elif output_type == 'image':
        return os.path.join(default_storage.location, "image_code.py")
    elif output_type == 'pdf':
        return os.path.join(default_storage.location, "pdf_code.py")
    else:
        return os.path.join(default_storage.location, "code.py")

#excute code on real data    
def process_csv_output(generated_text, csv_path=None, image_path=None):
    """
    Use real uploaded CSV (csv_path) + python_code extracted from LLM output to produce CSV results.
    We ignore inline CSV text from the model. SAS code is optional/back-compat only.
    """
    # 解析 LLM 输出（保留旧签名）——但不再使用 csv_data
    csv_data, sas_code, python_code = process_prompt(generated_text)

    # 基本校验：必须有真实 csv_path
    if not csv_path:
        logger.error("process_csv_output: csv_path is required for real-data execution.")
        return HttpResponse("csv_path is required", status=400)

    # python 代码必须存在
    if not python_code or not python_code.strip():
        logger.error("process_csv_output: no python code found in generated_text.")
        return HttpResponse("No python code found in the model output.", status=400)

    # 仅在存在时保存（为兼容或调试保留），但执行时不使用它
    if sas_code:
        _ = save_to_file("code.sas", sas_code)  # 可忽略返回值；不参与执行

    # 保存 python 代码到临时文件（供执行器读取）
    py_file_path = save_to_file("code.py", python_code)

    # 执行时 **传入真实 csv_path**（第一个位置参数 + 显式 kw）
    try:
        result = execute_python_code(
            csv_file_path=csv_path,            # 真实数据路径（storage key）
            py_file_path=py_file_path,
            output_type='csv',
            csv_path=csv_path,
            image_path=image_path
        )
    except Exception as e:
        logger.exception("process_csv_output: exception while executing python code.")
        return HttpResponse(str(e), status=500)

    if isinstance(result, Exception):
        logger.error(f"process_csv_output: execution error: {result}")
        return HttpResponse(str(result), status=500)

    # 为了兼容你原来的行为，仍返回模型原文（代码/说明），真正结果已被缓存用于预览/下载
    return HttpResponse(generated_text, content_type='text/plain')

def process_image_output(generated_text, csv_path=None, image_path=None):
    """
    Use real uploaded CSV (csv_path) + python_code extracted for image generation.
    The code must set image_data (base64). We ignore inline CSV text from the model.
    """
    # 解析 LLM 输出（保留旧签名）——但不再使用 csv_data
    csv_data, python_code = process_image_prompt(generated_text)

    # 基本校验
    if not csv_path:
        logger.error("process_image_output: csv_path is required for real-data execution.")
        return HttpResponse("csv_path is required", status=400)

    if not python_code or not python_code.strip():
        logger.error("process_image_output: no python code found in generated_text.")
        return HttpResponse("No python code found in the model output.", status=400)

    # 保存 python 代码到临时文件（供执行器读取）
    py_file_path = save_to_file("image_code.py", python_code)

    # 执行：传入真实 csv_path；期望代码里产出 image_data（base64）
    try:
        result = execute_python_code(
            csv_file_path=csv_path,            # 真实数据路径（storage key）
            py_file_path=py_file_path,
            output_type='image',
            csv_path=csv_path,
            image_path=image_path
        )
    except Exception as e:
        logger.exception("process_image_output: exception while executing python code.")
        return HttpResponse(str(e), status=500)

    if isinstance(result, Exception):
        logger.error(f"process_image_output: image generation error: {result}")
        return HttpResponse(str(result), status=500)

    return HttpResponse(generated_text, content_type='text/plain')

# =========================
# 8) Preview & validation
# =========================

def precompile_or_error(code: str, virtual_name: str = 'agentA_code.py'):
    try:
        compile(code, virtual_name, 'exec')
        return True, None
    except SyntaxError as e:
        line = (e.text or '').rstrip('\n')
        msg = f"SyntaxError at line {e.lineno}: {e.msg}. Source: {line}"
        return False, msg
    
def preview_csv_result(result_data, result_id):
    """Preview CSV result data."""
    try:
        df = pd.read_csv(StringIO(result_data))
        preview_info = {
            'success': True,
            'result_id': result_id,
            'output_type': 'csv',
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'sample_data': df.head(10).fillna('').to_dict('records'),
            'data_types': df.dtypes.astype(str).to_dict(),
            'file_size': len(result_data)
        }
        return JsonResponse(preview_info)
    except Exception as e:
        return JsonResponse({
            'success': True,
            'result_id': result_id,
            'output_type': 'csv',
            'raw_preview': result_data[:2000] + '...' if len(result_data) > 2000 else result_data,
            'file_size': len(result_data)
        })
    
def preview_image_result(result_data, result_id):
    """Preview image result."""
    # result_data should be base64 encoded image
    preview_info = {
        'success': True,
        'result_id': result_id,
        'output_type': 'image',
        'image_data': result_data,  # base64 encoded image
        'message': 'Image generated successfully'
    }
    return JsonResponse(preview_info)

def preview_pdf_result(result_data, result_id):
    """Preview PDF result."""
    preview_info = {
        'success': True,
        'result_id': result_id,
        'output_type': 'pdf',
        'file_size': len(result_data),
        'message': 'PDF report generated successfully'
    }
    return JsonResponse(preview_info)

# 在 gpt_backend_utils.py 中添加以下函数

import os
import mimetypes
import logging

logger = logging.getLogger(__name__)

# 文件类型分类
FILE_CATEGORIES = {
    'csv': ['.csv'],
    'text': ['.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h', '.json', 
             '.xml', '.html', '.css', '.sql', '.sh', '.yaml', '.yml', '.log',
             '.r', '.m', '.ipynb', '.tex', '.rst', '.conf', '.ini'],
    'image': ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.svg'],
    'pdf': ['.pdf'],
    'document': ['.doc', '.docx', '.odt', '.rtf']
}


def detect_file_type(filename: str) -> str:
    """
    根据文件扩展名检测文件类型
    
    Args:
        filename: 文件名
        
    Returns:
        文件类型: 'csv' | 'text' | 'image' | 'pdf' | 'document' | 'unknown'
    """
    if not filename:
        return 'unknown'
    
    ext = os.path.splitext(filename.lower())[1]
    
    for category, extensions in FILE_CATEGORIES.items():
        if ext in extensions:
            return category
    
    return 'unknown'


def read_text_file(file_path: str, max_chars: int = 50000) -> dict:
    """
    读取文本文件内容
    
    Args:
        file_path: 文件路径
        max_chars: 最大字符数限制
        
    Returns:
        dict: {
            'success': bool,
            'content': str,  # 文件内容
            'filename': str,
            'size': int,     # 字符数
            'truncated': bool,
            'error': str (如果失败)
        }
    """
    from django.core.files.storage import default_storage
    
    try:
        if not default_storage.exists(file_path):
            return {'success': False, 'error': f'File not found: {file_path}'}
        
        # 尝试多种编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1']
        content = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                with default_storage.open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if content is None:
            return {'success': False, 'error': 'Failed to decode file with supported encodings'}
        
        # 截断过长内容
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]
        
        filename = os.path.basename(file_path)
        
        return {
            'success': True,
            'content': content,
            'filename': filename,
            'size': len(content),
            'truncated': truncated,
            'encoding': used_encoding
        }
        
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return {'success': False, 'error': str(e)}


def analyze_text_with_gpt(text_content: str, user_prompt: str, api_key: str, 
                          filename: str = None) -> dict:
    """
    使用GPT分析文本内容（不执行代码）
    
    Args:
        text_content: 文本内容
        user_prompt: 用户需求
        api_key: OpenAI API key
        filename: 文件名（可选）
        
    Returns:
        dict: {
            'success': bool,
            'response': str,      # GPT的分析结果
            'has_code': bool,     # 结果中是否包含代码块
            'code_blocks': list,  # 提取的代码块列表
            'error': str (如果失败)
        }
    """
    import requests
    import re
    
    try:
        # 构建系统提示
        system_msg = (
            "You are a helpful assistant that analyzes text and code. "
            "Provide clear, insightful analysis based on the user's request. "
            "If you generate code, wrap it in ```language blocks for clarity. "
            "Do not execute code - only provide analysis and suggestions."
        )
        
        # 构建用户消息
        file_info = f"\n\n### Uploaded File: {filename}\n" if filename else ""
        user_content = (
            f"### User Request\n{user_prompt}\n"
            f"{file_info}"
            f"### File Content\n```\n{text_content}\n```\n\n"
            "Please analyze the content and respond to the user's request."
        )
        
        # 调用OpenAI API
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        logger.info("Calling OpenAI API for text analysis...")
        resp = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers, 
            json=payload, 
            timeout=120
        )
        
        if resp.status_code != 200:
            logger.error(f"OpenAI API error: {resp.status_code} - {resp.text}")
            return {
                'success': False,
                'error': f'OpenAI API error: {resp.status_code}'
            }
        
        content = resp.json()
        response_text = content.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        # 提取代码块
        code_pattern = r'```(\w+)?\n(.*?)```'
        code_blocks = []
        for match in re.finditer(code_pattern, response_text, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            code_blocks.append({
                'language': language,
                'code': code
            })
        
        return {
            'success': True,
            'response': response_text,
            'has_code': len(code_blocks) > 0,
            'code_blocks': code_blocks
        }
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timeout'}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'API request failed: {str(e)}'}
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        return {'success': False, 'error': str(e)}


def extract_code_for_execution(response_text: str, target_language: str = 'python') -> str:
    """
    从GPT响应中提取指定语言的代码（用于可选的代码执行）
    
    Args:
        response_text: GPT响应文本
        target_language: 目标语言（默认python）
        
    Returns:
        提取的代码字符串，如果没有找到则返回空字符串
    """
    import re
    
    # 匹配代码块
    pattern = rf'```{target_language}\n(.*?)```'
    matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # 返回第一个匹配的代码块
        return matches[0].strip()
    
    # 如果没有明确的语言标记，尝试匹配任意代码块
    pattern = r'```\n(.*?)```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return ''