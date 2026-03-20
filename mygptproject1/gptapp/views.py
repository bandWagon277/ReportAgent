# views.py（顶部）
from .gpt_backend_utils import (
    # 执行/环境
    execute_python_code, setup_execution_environment, handle_execution_result,
    # LLM I/O
    call_openai_chat, extract_python_code,
    build_agentA_messages, build_agentB_messages, build_enhanced_agentB_messages,
    get_prompt_path, get_pdf_dual_prompts, generate_agent_a_code,
    # CSV 摘要
    summarize_csv_for_prompt, summarize_csv_privacy,
    # 产物与组版
    persist_artifacts_and_build_manifest, execute_composer_code, build_artifacts_zip,
    # 输出辅助
    process_csv_output, process_image_output, get_script_path,
    # 预览/校验
    preview_csv_result, preview_image_result, preview_pdf_result, precompile_or_error,
    # 杂项
    sanitize_python, save_to_file, to_json_safe,
    # 文本pipline
    #detect_file_kind, is_text_extension, read_uploaded_text,analyze_general_text,
    #给定测试数据集
    #get_sample_csv_path,run_python_code_on_csv,
)
import json
import logging
import os
import requests
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import io
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import subprocess
import traceback
from django.core.files.storage import FileSystemStorage
from django.core.cache import cache
import uuid
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from PIL import Image
import re, textwrap, unicodedata
import tempfile
from datetime import datetime


# Define logger
logger = logging.getLogger(__name__)

# Define output types
OUTPUT_TYPES = {
    'CSV': 'csv',
    'IMAGE': 'image', 
    'PDF': 'pdf'
}


@csrf_exempt
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

            # Basic validation
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

            # ======= PDF: run full dual-agent pipeline and return result_id =======
            if output_type == 'PDF':
                try:
                    privacy_summary = summarize_csv_privacy(csv_path)
                except Exception as e:
                    logger.error(f"Error generating privacy summary: {e}")
                    return JsonResponse({'error': f'Error summarizing CSV file: {str(e)}'}, status=500)

                tmpl_A, tmpl_B = get_pdf_dual_prompts()

                # 1) 先生成 Agent A 代码（可选：一并返回给前端）
                try:
                    agent_a_code = generate_agent_a_code(
                        user_prompt=user_prompt,
                        privacy_summary=privacy_summary,
                        instruction_template_A=tmpl_A,
                        api_key=api_key
                    )
                except Exception as e:
                    logger.error(f"Error generating Agent A code: {e}")
                    return JsonResponse({'error': f'Error generating Agent A code: {str(e)}'}, status=500)

                # 2) 直接跑完整的 PDF 流水线（Agent A 执行 + 产物落盘 + Agent B 组版）
                return process_pdf_output(
                    generated_text=agent_a_code,
                    csv_path=csv_path,
                    image_path=image_path,
                    api_key=api_key,
                    privacy_summary=privacy_summary,
                    user_prompt=user_prompt,
                    instruction_template_A=tmpl_A,
                    instruction_template_B=tmpl_B
                )

            # ======= CSV / IMAGE original flow preserved =======
            # Read prompt file (kept for CSV/IMAGE original flow)
            prompt_path = get_prompt_path(output_type)
            logger.info(f"Using prompt file: {prompt_path}")

            prompt_text = ""
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                logger.info(f"Prompt file loaded successfully, length: {len(prompt_text)}")
            except Exception as e:
                logger.warning(f"Prompt file not found: {prompt_path} ({e})")

            # Build data_summary only for non-PDF (old behavior)
            try:
                data_summary = summarize_csv_for_prompt(csv_path, max_rows=20, max_cols=60)
                logger.info(f"Data summary generated, length: {len(data_summary)}")
            except Exception as e:
                logger.error(f"Error generating data summary: {e}")
                return JsonResponse({'error': f'Error reading CSV file: {str(e)}'}, status=500)

            if output_type in ('CSV', 'IMAGE'):
                system_msg = (
                    "You are a senior data analyst and Python author. "
                    "Generate robust Python code that respects the contract described by the user message."
                    "No placeholders. No fabricated data. Use VALIDATION_MODE exactly as specified. No internet."
                )
                user_content = (
                    f"{prompt_text}\n\n"
                    "### User Requirements\n"
                    f"{user_prompt}\n\n"
                    "### Data Context from CSV\n"
                    f"{data_summary}\n"
                    "### Implementation Notes\n"
                    "- The runtime environment already provides df/csv_path/image_path/VALIDATION_MODE.\n"
                    "- Follow the output contract and encoding requirements in the hint file.\n"
                )

                headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
                payload = {"model": "gpt-4o", "messages": [{"role": "system", "content": system_msg},
                                                           {"role": "user", "content": user_content}],
                           "temperature": 0.2}

                logger.info("Sending request to OpenAI API...")
                try:
                    resp = requests.post('https://api.openai.com/v1/chat/completions',
                                         headers=headers, json=payload, timeout=120)
                    logger.info(f"OpenAI API response status: {resp.status_code}")
                    if resp.status_code != 200:
                        logger.error(f"OpenAI API error: {resp.status_code} - {resp.text}")
                        return JsonResponse({'error': f'OpenAI API error: {resp.status_code}',
                                             'details': resp.text[:500]}, status=resp.status_code)

                    content = resp.json()
                    generated_text = content.get('choices', [{}])[0].get('message', {}).get('content', '').strip().replace('\\', '')
                    logger.info(f"Generated text length: {len(generated_text)}")
                    logger.info(f"Generated text preview: {generated_text[:800]}...")

                    if output_type == 'CSV':
                        return process_csv_output(generated_text, csv_path=csv_path, image_path=image_path)
                    else:
                        return process_image_output(generated_text, csv_path=csv_path, image_path=image_path)

                except requests.exceptions.Timeout:
                    logger.error("OpenAI API request timeout")
                    return JsonResponse({'error': 'Request timeout. Please try again.'}, status=504)
                except requests.exceptions.ConnectionError:
                    logger.error("OpenAI API connection error")
                    return JsonResponse({'error': 'Unable to connect to OpenAI API'}, status=503)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request to OpenAI API failed: {e}")
                    return JsonResponse({'error': f'API request failed: {str(e)}'}, status=500)

            # Fallback (should not reach here)
            return JsonResponse({'error': 'Unsupported output type'}, status=400)

        except Exception as e:
            logger.error(f"Unexpected error in render_gpt_interface: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return JsonResponse({'error': f'Internal server error: {str(e)}',
                                 'traceback': traceback.format_exc()}, status=500)

    # GET
    return render(request, 'gpt_interface.html')


@csrf_exempt
def execute_pdf_pipeline(request):
    """
    一次性跑 Agent A(执行) + Agent B(组版)，生成 PDF。
    前端会传：csv_path, image_path(可选), user_prompt, agentA_code 或 agentA_code_path(二选一)
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body or '{}')
    except Exception as e:
        return JsonResponse({'error': f'Invalid JSON: {e}'}, status=400)

    csv_path        = data.get('csv_path')
    image_path      = data.get('image_path')
    user_prompt     = data.get('user_prompt', '')
    agentA_code     = data.get('agentA_code')
    agentA_code_path= data.get('agentA_code_path')

    if not csv_path:
        return JsonResponse({'error': 'csv_path is required'}, status=400)
    if not (agentA_code or agentA_code_path):
        # 允许后端自行再调用 A 生成，但两阶段设计下通常应至少给一个
        logger.warning("No agentA_code provided; will fall back to re-generating Agent A code.")
    
    # 若只给了路径，读取代码文本
    if (not agentA_code) and agentA_code_path:
        try:
            with default_storage.open(agentA_code_path, 'r') as f:
                agentA_code = f.read()
        except Exception as e:
            logger.exception("Failed to read agentA_code_path")
            return JsonResponse({'error': f'Cannot read agentA_code_path: {e}'}, status=500)

    # 取 API key
    api_key = getattr(settings, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')

    # 取模板（如果你的项目里有现成函数就用它；没有可设为 None）
    try:
        tmpl_A, tmpl_B = get_pdf_dual_prompts()  # 若无此函数，设为: tmpl_A = tmpl_B = None
    except Exception:
        tmpl_A = tmpl_B = None

    # 调用你现有的 pipeline 实现（见 B 部分的“微调”）
    return process_pdf_output(
        generated_text=agentA_code,          # 关键：把前端的 A 代码传进去
        csv_path=csv_path,
        image_path=image_path,
        api_key=api_key,
        privacy_summary=None,                # 函数内部会生成或你也可提前生成
        user_prompt=user_prompt,
        instruction_template_A=tmpl_A,
        instruction_template_B=tmpl_B
    )


def process_pdf_output(generated_text: str | None,
                       csv_path: str | None = None,
                       image_path: str | None = None,
                       api_key: str | None = None,
                       privacy_summary: dict | None = None,
                       user_prompt: str | None = None,
                       instruction_template_A: str | None = None,
                       instruction_template_B: str | None = None):
    """
    Enhanced two-agent pipeline with statistical data transmission:
      Agent A: codegen -> analysis on real df -> artifacts with data_summary -> manifest built
      Agent B: composer -> render PDF from manifest with statistical insights

    Returns JsonResponse with {success, result_id, output_type, preview_data, manifest_id, narrative_excerpt}
    """
    if not api_key:
        return JsonResponse({'error': 'OpenAI API key not configured'}, status=500)
    if not csv_path:
        return JsonResponse({'error': 'csv_path is required'}, status=400)

    # Ensure privacy summary
    try:
        privacy_summary = privacy_summary or summarize_csv_privacy(csv_path)
    except Exception as e:
        logger.error(f"Failed to build privacy summary: {e}")
        return JsonResponse({'error': f'Failed to summarize CSV: {str(e)}'}, status=500)

    # --- Agent A: codegen (artifacts producer with statistical summaries) ---
    try:
        if generated_text:   # 前端传了 agentA_code
            code_A = generated_text
            codeA_path = save_to_file("agentA_code.py", code_A)
            logger.info("Using Agent A code provided by frontend, skip API call.")
        else:
            messages_A = build_agentA_messages(
                user_prompt or "Generate a clinical report.",
                privacy_summary,
                instruction_template_A
            )
            logger.info("Calling Agent A (enhanced codegen)...")
            resp_A = call_openai_chat(api_key, messages_A)
            code_A = extract_python_code(resp_A)
            if not code_A:
                return JsonResponse({'error': 'Agent A did not return Python code.'}, status=400)
            codeA_path = save_to_file("agentA_code.py", code_A)
    except Exception as e:
        logger.error(f"Agent A call failed: {e}")
        return JsonResponse({'error': f'Agent A error: {str(e)}'}, status=500)

    # Execute Agent A code to get enhanced artifacts with data_summary
    artifacts = execute_python_code(
        csv_file_path=csv_path,
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

    # Persist artifacts -> manifest (images stored, data_summary kept in manifest)
    try:
        manifest_id, manifest = persist_artifacts_and_build_manifest(artifacts)
    except Exception as e:
        logger.error(f"Persist artifacts error: {e}")
        return JsonResponse({'error': f'Persist artifacts error: {str(e)}'}, status=500)

    # --- Agent B: composer (PDF writer with statistical insights) ---
    try:
        # Build enhanced messages with statistical data for Agent B
        messages_B = build_enhanced_agentB_messages(manifest, instruction_template_B)
        logger.info("Calling Agent B (enhanced composer with statistical data)...")
        resp_B = call_openai_chat(api_key, messages_B)
        code_B = extract_python_code(resp_B)
        if not code_B:
            return JsonResponse({'error': 'Agent B did not return Python code.'}, status=400)
        codeB_path = save_to_file("agentB_composer.py", code_B)
    except Exception as e:
        logger.error(f"Agent B call failed: {e}")
        return JsonResponse({'error': f'Agent B error: {str(e)}'}, status=500)

    # Execute composer with enhanced manifest
    try:
        manifest_paths = cache.get(f"manifest:{manifest_id}:paths") or {"figures": {}, "tables": {}}
        pdf_bytes, narrative_text = execute_composer_code(codeB_path, manifest, manifest_paths)
    except Exception as e:
        logger.error(f"Composer execution error: {e}")
        return JsonResponse({'error': f'Composer execution error: {str(e)}'}, status=500)

    # Cache final PDF for download/preview
    result_id = str(uuid.uuid4())
    cache.set(f"result_{result_id}", pdf_bytes, timeout=3600)

    return JsonResponse({
        'success': True,
        'result_id': result_id,
        'output_type': 'pdf',
        'manifest_id': manifest_id,
        'preview_data': 'PDF report generated successfully',
        'agentA_code': code_A,
        'narrative_excerpt': (narrative_text[:400] + '...') if narrative_text and len(narrative_text) > 400 else (narrative_text or "")
    }, status=200)







@csrf_exempt
def upload_csv(request):
    """Upload a CSV file and return its stored path."""
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        if csv_file:
            file_path = default_storage.save(f'uploads/{csv_file.name}', csv_file)
            return JsonResponse({'file_path': file_path}, status=200)
        else:
            return JsonResponse({'error': 'No CSV file provided'}, status=400)
    else:
        return HttpResponse('Method not allowed', status=405)


# 确保文件顶部导入了必要的模块
# from django.http import HttpResponse, JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import traceback
# from django.conf import settings
# import logging
# logger = logging.getLogger(__name__)

@csrf_exempt
def upload_and_analyze_text(request):
    """
    健壮版本：处理文件上传和自动分流。
    """
    # 1. 优先检查请求方法，确保非POST请求不会进入复杂逻辑
    if request.method != 'POST':
        # 记录警告信息
        logger.warning("Method not allowed: %s requested for /upload_and_analyze_text/", request.method)
        return HttpResponse('Method not allowed', status=405)

    # 2. 初始化 ctx 变量，它将被用于日志记录和异常返回
    # 由于该变量位于 try 块之外，必须确保其初始化是绝对安全的。
    ctx = {
        "method": request.method,
        "headers_sample": {}, # 初始为空，稍后在 try 块中安全填充
        "has_files": False,
        "file_name": "N/A",
        "file_ct": "N/A",
        "prompt_len": 0,
        "detected_file_type": "N/A",
    }
    
    # 3. 顶级 try...except 块：捕获所有可能导致 500 的异常
    try:
        # 3.1. 安全地填充 ctx 变量
        ctx["headers_sample"] = {k: request.META.get(k) for k in [
            "CONTENT_TYPE", "CONTENT_LENGTH", "HTTP_USER_AGENT", "HTTP_ORIGIN", "REMOTE_ADDR"
        ]}
        ctx["has_files"] = bool(getattr(request, "FILES", None) and request.FILES)

        # 3.2. API Key 检查 (如果缺失，直接返回 500 JsonResponse)
        api_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not api_key:
            logger.error("OPENAI_API_KEY is not configured in settings.")
            return JsonResponse({'error': 'OPENAI_API_KEY not configured'}, status=500)
            
        # 3.3. 获取文件和 prompt
        upload_file = request.FILES.get('file')
        user_prompt = (request.POST.get('prompt') or '').strip()

        ctx["prompt_len"] = len(user_prompt)
        
        if not upload_file:
            logger.error("upload_and_analyze_text: No file provided; ctx=%s", ctx)
            return JsonResponse({'error': 'No file provided (expecting FormData key "file")'}, status=400)

        ctx["file_name"] = getattr(upload_file, "name", None)
        ctx["file_ct"] = getattr(upload_file, "content_type", None)

        # 3.4. 类型判定 (这里保持您原有的 try...except，但捕获后使用 logger.exception)
        try:
            # 假设 detect_file_kind 是一个可能出错的外部调用
            file_type = detect_file_kind(ctx["file_name"], ctx["file_ct"])
            ctx["detected_file_type"] = file_type
        except Exception:
            # 使用 logger.exception 自动记录完整 Traceback
            logger.exception("detect_file_kind failed; ctx=%s", ctx)
            if settings.DEBUG:
                # 重新抛出异常以获取 traceback，或者使用 traceback.format_exc()
                tb = traceback.format_exc()
                return JsonResponse({'error': 'detect_file_kind failed', 'traceback': tb, 'ctx': ctx}, status=500)
            return JsonResponse({'error': 'detect_file_kind failed'}, status=500)

        # 3.5. CSV 和 TEXT 分支逻辑 (保持不变，因为它们内部已有捕获)
        
        # ---- CSV：仅保存路径，走原有 CSV 流程 ----
        if file_type == 'csv':
            try:
                # ... CSV 保存逻辑 ...
                path = default_storage.save(f'uploads/{upload_file.name}', upload_file)
                return JsonResponse({'file_type': 'csv', 'csv_path': path}, status=200)
            except Exception:
                logger.exception("Saving CSV failed; ctx=%s", ctx)
                # ... 错误返回逻辑 ...

        # ---- TEXT：读取 + 调 GPT ----
        if file_type == 'text':
            if not user_prompt: # 如果你希望允许空 prompt，可以去掉此校验
                return JsonResponse({'error': 'Missing "prompt" for text analysis'}, status=400)
            
            try:
                # 假设 read_uploaded_text 是一个可能出错的外部调用
                text_content = read_uploaded_text(upload_file, max_bytes=200_000)
            except Exception:
                logger.exception("read_uploaded_text failed; ctx=%s", ctx)
                # ... 错误返回逻辑 ...

            try:
                # 假设 analyze_general_text 是一个可能出错的外部调用
                result = analyze_general_text(
                    user_prompt=user_prompt,
                    text=text_content,
                    api_key=api_key,
                    model='gpt-4o-mini',
                    temperature=0.3
                )
                return JsonResponse({'file_type': 'text', **result}, status=200)
            except Exception as e:
                logger.exception("analyze_general_text failed; ctx=%s", ctx)
                # ... 错误返回逻辑 ...

        # ---- 其它：不支持 ----
        logger.warning("Unsupported file type: %s; ctx=%s", file_type, ctx)
        return JsonResponse(
            {'error': f'Unsupported file type for general analysis: {ctx["file_name"]}'},
            status=400
        )

    except Exception as e:
        # 4. 兜底：处理所有未被内部捕获的顶级异常（包括 ctx 初始化失败）
        # 必须使用 logger.exception 记录，因为这是导致 500 的根本原因
        logger.exception("upload_and_analyze_text TOP-LEVEL FAILURE; ctx=%s", ctx)
        
        payload = {'error': 'A severe internal error occurred: ' + str(e)}
        if settings.DEBUG:
             # 如果 settings.DEBUG 为 True，返回详细信息以供调试
            tb = traceback.format_exc()
            payload.update({'traceback': tb, 'ctx': ctx})
            
        return JsonResponse(payload, status=500)

@csrf_exempt
def upload_files(request):
    """
    Upload CSV (required) and optional IMAGE file.
    Return their stored paths: {'csv_path': ..., 'image_path': ... (optional)}
    """
    if request.method != 'POST':
        return HttpResponse('Method not allowed', status=405)

    csv_file = request.FILES.get('csv_file')
    image_file = request.FILES.get('image_file')

    if not csv_file:
        return JsonResponse({'error': 'No CSV file provided'}, status=400)

    try:
        csv_path = default_storage.save(f'uploads/{csv_file.name}', csv_file)
        resp = {'csv_path': csv_path}

        if image_file:
            # 仅做基本类型过滤，可按需扩展
            valid_images = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
            if not image_file.name.lower().endswith(valid_images):
                return JsonResponse({'error': 'Invalid image file type'}, status=400)
            image_path = default_storage.save(f'uploads/{image_file.name}', image_file)
            resp['image_path'] = image_path

        return JsonResponse(resp, status=200)
    except Exception as e:
        logger.error(f"upload_files error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def process_csv(request):
    """Process CSV with different output types (supports dry-run validation)."""
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))

        # 路径与类型
        csv_file_path = data.get('file_path', '')     # 你的前端传的是 file_path；我们同时当作 csv_path 用
        image_path = data.get('image_path')
        output_type = (data.get('output_type', 'csv') or 'csv').lower()  # 'csv' | 'image' | 'pdf'

        # 新增：dry-run & row_limit
        dry_run = bool(data.get('dry_run', False))
        row_limit = data.get('row_limit')
        try:
            row_limit = int(row_limit) if row_limit is not None else None
        except Exception:
            row_limit = None

        script_path = get_script_path(output_type)

        try:
            # 只调用一次；统一把 csv_path / image_path / dry_run / row_limit 传入
            results = execute_python_code(
                csv_file_path=csv_file_path,    # 兼容旧参数；下面也通过 csv_path 传入
                py_file_path=script_path,
                output_type=output_type,
                csv_path=csv_file_path,
                image_path=image_path,
                dry_run=dry_run,
                row_limit=row_limit
            )

            # dry-run：只返回校验结果，不落缓存、不生成 result_id
            if dry_run:
                if isinstance(results, dict) and results.get('ok'):
                    return JsonResponse({'ok': True})
                # 失败时，results 可能是 {'ok': False, 'error': ...} 或 Exception
                err_msg = results.get('error') if isinstance(results, dict) else str(results)
                return JsonResponse({'ok': False, 'error': err_msg}, status=500)

            # 正式执行：必须不是 Exception
            if isinstance(results, Exception):
                return JsonResponse({'error': str(results)}, status=500)

            # 生成 ID 并缓存
            result_id = str(uuid.uuid4())

            if output_type == 'csv':
                cache.set(f'result_{result_id}', results, timeout=3600)  # results 为 CSV 文本
                preview_data = results[:1000] + '...' if len(results) > 1000 else results

            elif output_type == 'image':
                cache.set(f'result_{result_id}', results, timeout=3600)  # base64 或 bytes（你下游按 base64 解）
                preview_data = "Image generated successfully"

            elif output_type == 'pdf':
                cache.set(f'result_{result_id}', results, timeout=3600)  # PDF bytes
                preview_data = "PDF report generated successfully"

            else:
                return JsonResponse({'error': 'Invalid output type'}, status=400)

            return JsonResponse({
                'success': True,
                'result_id': result_id,
                'output_type': output_type,
                'preview_data': preview_data
            })

        except Exception as e:
            logger.error(f"Error processing {output_type}: {e}")
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'gpt_interface.html')


@csrf_exempt
def process_text(request):
    """
    不上传文件，直接提交纯文本字符串 + 用户需求。
    Body(JSON): { "text": "...", "prompt": "..." }
    """
    if request.method != 'POST':
        return HttpResponse('Method not allowed', status=405)

    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        return JsonResponse({'error': 'OPENAI_API_KEY not configured'}, status=500)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    text = (data.get('text') or '').strip()
    user_prompt = (data.get('prompt') or '').strip()

    if not text:
        return JsonResponse({'error': 'No "text" provided'}, status=400)
    if not user_prompt:
        return JsonResponse({'error': 'No "prompt" provided'}, status=400)

    try:
        result = analyze_general_text(
            user_prompt=user_prompt,
            text=text,
            api_key=api_key,
            model='gpt-4o-mini',
            temperature=0.3
        )
        return JsonResponse(result, status=200)
    except Exception as e:
        logger.exception("process_text failed")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def preview_result(request):
    """Preview processing results based on output type."""
    if request.method == 'GET':
        result_id = request.GET.get('result_id')
        output_type = request.GET.get('output_type', 'csv')
        
        if not result_id:
            return JsonResponse({'error': 'No result ID provided'}, status=400)
        
        result_data = cache.get(f'result_{result_id}')
        if not result_data:
            return JsonResponse({'error': 'Result not found or expired'}, status=404)
        
        try:
            if output_type == 'csv':
                return preview_csv_result(result_data, result_id)
            elif output_type == 'image':
                return preview_image_result(result_data, result_id)
            elif output_type == 'pdf':
                return preview_pdf_result(result_data, result_id)
            else:
                return JsonResponse({'error': 'Invalid output type'}, status=400)
                
        except Exception as e:
            logger.error(f"Error previewing {output_type} result: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def save_agent_a_code(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        payload = json.loads(request.body or '{}')
    except Exception as e:
        return JsonResponse({'error': f'Invalid JSON: {e}'}, status=400)

    code = payload.get('code') or ''
    code_path = payload.get('code_path')  # 可选：覆盖保存路径

    code = sanitize_python(code)
    ok, err = precompile_or_error(code, 'agentA_code.py')
    if not ok:
        return JsonResponse({'error': f'Invalid Python code: {err}'}, status=400)

    # 选择保存策略：覆盖 or 版本化，这里给两种：
    version = datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    if not code_path:
        # 初次保存或不想覆盖：创建新文件（版本化）
        file_name = f'agentA_code_{version.replace(":","-")}.py'
        file_path = os.path.join(default_storage.location, file_name)
    else:
        # 覆盖已有文件
        file_path = code_path

    with default_storage.open(file_path, 'w') as f:
        f.write(code)

    if not default_storage.exists(file_path):
        return JsonResponse({'error': 'Failed to save code'}, status=500)

    return JsonResponse({'success': True, 'code_path': file_path, 'version': version}, status=200)

@csrf_exempt
def download_result(request):
    """Download processing results based on type."""
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    result_id = request.GET.get('result_id')
    output_type_raw = request.GET.get('output_type', 'csv')

    if not result_id:
        return JsonResponse({'error': 'No result ID provided'}, status=400)

    # --- 新增：归一化 output_type & 别名映射 ---
    alias_map = {
        'csv': 'csv',
        'image': 'image', 'img': 'image', 'png': 'image', 'jpg': 'image', 'jpeg': 'image',
        'pdf': 'pdf', 'report': 'pdf'
    }
    key = (output_type_raw or '').strip().lower()
    output_type = alias_map.get(key, 'csv')
    # ----------------------------------------

    result_data = cache.get(f'result_{result_id}')
    if result_data is None:
        return JsonResponse({'error': 'Result not found or expired'}, status=404)

    try:
        if output_type == 'csv':
            # CSV 可能是 str 或 bytes
            if isinstance(result_data, (bytes, bytearray)):
                content = result_data
            else:
                content = (result_data or '').encode('utf-8')

            response = HttpResponse(content, content_type='text/csv; charset=utf-8')
            response['Content-Disposition'] = f'attachment; filename="processed_data_{result_id[:8]}.csv"'

        elif output_type == 'image':
            # IMAGE 可能是 base64 或 bytes
            if isinstance(result_data, (bytes, bytearray)):
                image_bytes = result_data
            else:
                try:
                    image_bytes = base64.b64decode(result_data)
                except Exception as e:
                    logger.error(f"Image base64 decoding error: {e}")
                    return JsonResponse({'error': 'Invalid image data'}, status=500)

            # 这里默认 PNG；如果你的图片是 JPG，则把下面两行改为 image/jpeg 和 .jpg
            response = HttpResponse(image_bytes, content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="generated_chart_{result_id[:8]}.png"'

        elif output_type == 'pdf':
            # PDF 通常为 bytes；若不是，则尝试把 base64 转 bytes（容错）
            if isinstance(result_data, (bytes, bytearray)):
                pdf_bytes = result_data
            else:
                try:
                    pdf_bytes = base64.b64decode(result_data)
                except Exception:
                    logger.error("PDF result is not binary data nor base64")
                    return JsonResponse({'error': 'Invalid PDF data'}, status=500)

            response = HttpResponse(pdf_bytes, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="analysis_report_{result_id[:8]}.pdf"'

        else:
            return JsonResponse({'error': 'Invalid output type'}, status=400)

        # 防缓存，避免浏览器复用旧的 Content-Type
        response['Cache-Control'] = 'no-store'
        logger.info(f"Result {result_id} ({output_type}) downloaded successfully")
        return response

    except Exception as e:
        logger.error(f"Error downloading {output_type} result: {e}")
        return JsonResponse({'error': str(e)}, status=500)


import re


@csrf_exempt
def download_artifacts_zip(request):
    manifest_id = request.GET.get("manifest_id")
    if not manifest_id:
        return JsonResponse({'error': 'manifest_id is required'}, status=400)
    try:
        zip_bytes = build_artifacts_zip(manifest_id)
        resp = HttpResponse(zip_bytes, content_type='application/zip')
        resp['Content-Disposition'] = f'attachment; filename="artifacts_{manifest_id[:8]}.zip"'
        return resp
    except Exception as e:
        logger.error(f"ZIP build error: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
def execute_code_on_sample(request):
    if request.method != 'POST':
        return HttpResponse('Method not allowed', status=405)
    try:
        data = json.loads(request.body.decode('utf-8'))
        code = (data.get('code') or '').strip()
        lang = (data.get('language') or 'python').lower()
        if not code:
            return JsonResponse({'error': 'Missing code'}, status=400)
        if lang != 'python':
            return JsonResponse({'error': 'Only Python execution is supported for now'}, status=400)

        sample_csv = get_sample_csv_path()
        out = run_python_code_on_csv(code, sample_csv)
        return JsonResponse(out, status=200 if out.get('ok') else 500)
    except Exception as e:
        logger.exception("execute_code_on_sample failed")
        return JsonResponse({'error': str(e)}, status=500)

