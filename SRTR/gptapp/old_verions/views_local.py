import os
import json
import csv
import uuid
import logging
import textwrap
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from io import StringIO

import requests
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

logger = logging.getLogger(__name__)

# ==========
# Core Utilities
# ==========

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# if temperature higher, then the variablity of the answer is higher(prediction more random with much flat distribution)
def _openai_chat(messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """
    Minimal wrapper for OpenAI Chat Completions API.
    """
    url = f"{OPENAI_API_BASE}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            logger.error("OpenAI error %s: %s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"OpenAI API error: {resp.status_code}")
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.exception("OpenAI request failed: %s", e)
        raise

# ==========
# Page Rendering
# ==========

def render_gpt_interface(request):
    """Render main interface"""
    return render(request, "gpt_interface.html")

# ==========
# Path Configuration
# ==========

DATA_REPO = Path(r"C:/Users/18120/Desktop/OPENAIproj/SRTR/data_repo")
DICT_ROOT = DATA_REPO / "dictionaries"
DICT_INDEX = DATA_REPO / "meta" / "dictionaries.index.json"
CONCEPTS_DIR = DATA_REPO / "concepts"
DOCS_DIR = DATA_REPO / "docs"

# ==========
# Dictionary Index Building & Loading
# ==========

def build_dictionaries_index():
    """
    Scan data_repo/dictionaries/**.csv and read variable definitions.
    
    Important: These CSVs are data dictionary files with format:
    Variable, Type, Length, Format, Label
    CAN_ABO, Char, 2, $2., ABO blood type
    DON_AGE, Num, 8, 8., Donor age
    
    We need to read values from the first column as variable names!
    """
    DICT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    files = []
    variables_map = {}

    logger.info("Building dictionaries index...")
    
    for csv_path in DICT_ROOT.rglob("*.csv"):
        rel = csv_path.relative_to(DATA_REPO).as_posix()
        category = csv_path.parent.name  # e.g. General, Kidney_Pancreas
        headers = []
        variables_in_file = []

        try:
            # Try multiple encodings
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(csv_path, "r", encoding=encoding) as f:
                        content = f.read()
                        break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.warning(f"Could not decode {csv_path} with any encoding")
                continue
            
            # Parse CSV
            reader = csv.DictReader(StringIO(content))
            headers = reader.fieldnames or []
            
            logger.info(f"Reading {csv_path.name}, headers: {headers}")
            
            # Read all rows and extract values from "Variable" column
            if "Variable" in headers:
                row_count = 0
                for row in reader:
                    var_name = row.get("Variable", "").strip()
                    if var_name and var_name != "Variable":  # Avoid duplicate headers
                        variables_in_file.append(var_name)
                        row_count += 1
                logger.info(f"  Found {row_count} variables in {csv_path.name}")
            else:
                # If no Variable column, use headers as variable names (compatibility)
                variables_in_file = [h.strip() for h in headers if h.strip()]
                logger.info(f"  No 'Variable' column, using headers as variables: {variables_in_file}")
                    
        except Exception as e:
            logger.error(f"Failed to read {csv_path}: {e}", exc_info=True)
            headers = []
            variables_in_file = []

        file_path = str(csv_path).replace("\\", "/")
        files.append({
            "path": file_path,
            "name": csv_path.name,
            "category": category,
            "headers": headers,
            "variables": variables_in_file[:100],  # Preview first 100 variables
            "total_variables": len(variables_in_file)
        })

        # Build variable name to file path mapping
        for var in variables_in_file:
            var_upper = var.upper()
            variables_map.setdefault(var_upper, [])
            if file_path not in variables_map[var_upper]:
                variables_map[var_upper].append(file_path)

    index = {
        "files": files,
        "variables": variables_map,
        "build_time": str(Path(__file__).stat().st_mtime),
        "total_variables": len(variables_map)
    }
    
    with open(DICT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dictionary index built: {len(files)} files, {len(variables_map)} unique variables")
    for file_info in files:
        logger.info(f"  - {file_info['name']}: {file_info['total_variables']} variables")
    
    return index

def load_dictionaries_index():
    """Load dictionary index, build if not exists"""
    if not DICT_INDEX.exists():
        return build_dictionaries_index()
    
    try:
        with open(DICT_INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)
            # Check index version, rebuild if outdated
            if "total_variables" not in index:
                logger.info("Old index format detected, rebuilding...")
                return build_dictionaries_index()
            return index
    except Exception as e:
        logger.error(f"Failed to load index, rebuilding: {e}")
        return build_dictionaries_index()

def guess_paths_for_variable(var_name: str, index: dict, topk: int = 3) -> List[str]:
    """
    Given a variable name, return Top-K most likely dictionary file paths.
    Ranking logic:
      1) Exact variable match (priority)
      2) Partial variable name match
      3) Filename contains variable name
    """
    if not var_name:
        return []
    
    v = var_name.strip().upper()
    candidates = []

    # (1) Variable → file direct mapping (exact match)
    if v in index.get("variables", {}):
        for p in index["variables"][v]:
            candidates.append((p, 1.0, "exact_match"))
            logger.info(f"Found exact match for {v} in {p}")

    # (2) Partial match: find files containing this variable
    if not candidates:
        for f in index.get("files", []):
            score = 0.0
            match_type = ""
            
            # Check variable list in file
            for file_var in f.get("variables", []):
                if v == file_var.upper():
                    score = max(score, 1.0)
                    match_type = "exact"
                elif v in file_var.upper():
                    score = max(score, 0.8)
                    match_type = "partial"
                elif file_var.upper().startswith(v):
                    score = max(score, 0.7)
                    match_type = "prefix"
            
            # Filename clues
            if v in f["name"].upper():
                score = max(score, 0.5)
                match_type = match_type or "filename"
            
            if score > 0:
                candidates.append((f["path"], score, match_type))

    # Sort and deduplicate
    uniq = {}
    for p, s, t in candidates:
        if p not in uniq or uniq[p][0] < s:
            uniq[p] = (s, t)
    
    ranked = sorted(uniq.items(), key=lambda x: -x[1][0])
    result = [p for p, _ in ranked[:topk]]
    
    if result:
        logger.info(f"Found {len(result)} matches for '{var_name}': {result}")
    else:
        logger.warning(f"No matches found for '{var_name}'")
    
    return result

# ==========
# Initial Agent: Intelligent Path Planning
# ==========

def _initial_agent_plan_locally(user_query: str) -> dict:
    """
    Initial Agent: Intelligent Path Planning
    
    Responsibilities:
    1. Understand user intent (definition query vs concept calculation vs data analysis)
    2. Fuzzy matching and semantic understanding (not just exact matching)
    3. Multi-path recommendation (related variables, concepts, docs)
    4. Execution step planning
    """
    q = user_query.strip()
    idx = load_dictionaries_index()

    # ===== Step 1: Extract Keywords =====
    # Extract variable names (without word boundaries for Chinese compatibility)
    patterns = [
        r"[A-Z][A-Z0-9_]{2,}",
        r"(?<=[^A-Z])[A-Z][A-Z0-9_]{2,}(?=[^A-Z0-9_]|$)",
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, q)
        all_matches.extend(matches)
    
    seen = set()
    unique_matches = []
    for m in all_matches:
        if m not in seen:
            seen.add(m)
            unique_matches.append(m)
    
    var_token = None
    for match in unique_matches:
        if "_" in match:
            var_token = match
            break
    if not var_token and unique_matches:
        var_token = unique_matches[0]
    
    logger.info(f"Query: '{q}' | Extracted tokens: {unique_matches} | Primary variable: {var_token}")

    # ===== Step 2: Use LLM for intent understanding and path recommendation =====
    # This is the real "Agent" part!
    plan = _intelligent_path_planning(q, var_token, unique_matches, idx)
    
    return plan


def _intelligent_path_planning(query: str, primary_var: str, all_vars: list, index: dict) -> dict:
    """
    Use LLM for intelligent path planning
    
    This is the core of Initial Agent: not just character matching, but semantic understanding
    """
    
    # Basic matching first (fast path)
    dict_paths = []
    if primary_var:
        dict_paths = guess_paths_for_variable(primary_var, index, topk=3)
    
    # Detect concept keywords
    concept_paths = []
    concept_keywords = []
    is_concept_query = any([
        re.search(r"(code|calculate|formula|algorithm|compute)", query, re.I),
        re.search(r"\b(egfr|kdpi|epts|cpra)\b", query, re.I)
    ])
    
    if is_concept_query:
        concept_patterns = {
            "eGFR": r"\begfr\b",
            "KDPI": r"\bkdpi\b",
            "EPTS": r"\bepts\b",
            "CPRA": r"\bcpra\b"
        }
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, query, flags=re.I):
                concept_keywords.append(concept)
                concept_r = CONCEPTS_DIR / "concepts.R"
                if concept_r.exists() and str(concept_r) not in concept_paths:
                    concept_paths.append(str(concept_r))
    
    # If basic matching found results and it's a simple query, return directly (fast path)
    if dict_paths and not is_concept_query and len(query) < 50:
        intent = ["explain-variable"]
        rationale = f"Fast path: Exact match for '{primary_var}'"
        logger.info(f"Fast path taken: {rationale}")
    else:
        # ===== Intelligent path: Use LLM to understand complex queries =====
        try:
            intent, rationale, additional_suggestions = _ask_llm_for_intent(
                query, primary_var, all_vars, dict_paths, concept_keywords, index
            )
            
            # LLM may recommend additional variables or files
            if additional_suggestions:
                for sugg in additional_suggestions:
                    paths = guess_paths_for_variable(sugg, index, topk=2)
                    dict_paths.extend(paths)
                dict_paths = list(dict.fromkeys(dict_paths))[:5]  # Deduplicate, max 5
                
        except Exception as e:
            logger.warning(f"LLM intent analysis failed, using rule-based fallback: {e}")
            intent = ["explain-variable"] if dict_paths else ["general-query"]
            rationale = f"Rule-based: Variable={primary_var}, Concepts={concept_keywords}"
    
    # Document paths
    docs_paths = []
    doc_html = DOCS_DIR / "data_dictionary.html"
    if doc_html.exists():
        docs_paths.append(str(doc_html))
    
    return {
        "variable": primary_var,
        "all_variables": all_vars,
        "dictionaries": dict_paths,
        "concepts": concept_paths,
        "concept_keywords": concept_keywords,
        "docs": docs_paths,
        "data": [],
        "intent": intent,
        "rationale": rationale
    }


def _ask_llm_for_intent(query: str, primary_var: str, all_vars: list, 
                        found_paths: list, concepts: list, index: dict) -> Tuple[list, str, list]:
    """
    Use LLM to understand user intent and recommend related variables and paths
    
    This is the intelligent core of Initial Agent!
    """
    
    # Get variable samples from index (for LLM recommendations)
    available_vars = list(index.get("variables", {}).keys())[:100]
    
    system = textwrap.dedent("""
    You are the SRTR Data Analysis Path Planning Assistant (Initial Agent).
    
    Your tasks:
    1. Understand the user's real intent
    2. Identify what the user wants (variable definition, concept calculation, data analysis, etc.)
    3. Recommend related variable names (even if not explicitly mentioned by user)
    4. Judge query complexity
    
    Return JSON format:
    {
      "intent": ["explain-variable"|"explain-concept"|"data-analysis"|"general-query"],
      "rationale": "brief explanation of your judgment",
      "suggested_variables": ["VAR1", "VAR2"],  // additional related variables to recommend
      "complexity": "simple"|"moderate"|"complex"
    }
    
    Examples:
    Q: "What variables are related to donor age?"
    A: {
      "intent": ["explain-variable"],
      "rationale": "User wants to know variables related to donor age",
      "suggested_variables": ["DON_AGE", "DON_AGE_IN_MONTHS"],
      "complexity": "simple"
    }
    
    Q: "How to calculate kidney function?"
    A: {
      "intent": ["explain-concept", "explain-variable"],
      "rationale": "Needs eGFR concept calculation and related variables like CRE_SERUM",
      "suggested_variables": ["CRE_SERUM", "CAN_AGE"],
      "complexity": "moderate"
    }
    """).strip()
    
    user_content = {
        "query": query,
        "extracted_variable": primary_var,
        "all_extracted": all_vars,
        "found_paths_count": len(found_paths),
        "detected_concepts": concepts,
        "available_variables_sample": available_vars[:50]
    }
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
    ]
    
    raw = _openai_chat(messages, temperature=0.3, max_tokens=500)
    
    # Parse LLM response
    try:
        result = json.loads(raw)
        intent = result.get("intent", ["general-query"])
        rationale = result.get("rationale", "LLM analysis")
        suggestions = result.get("suggested_variables", [])
    except:
        intent = ["explain-variable"] if primary_var else ["general-query"]
        rationale = "LLM parsing failed, using fallback"
        suggestions = []
    
    return intent, rationale, suggestions

# ==========
# Agent A: Generate Answer
# ==========

def _agent_a_generate_answer(user_query: str, plan: dict) -> dict:
    """
    Generate answer based on Initial Agent's plan.
    For variable definitions: Read complete definition row from CSV.
    For concept queries: Read content from concept files.
    """
    variable = plan.get("variable")
    dict_paths = plan.get("dictionaries", [])[:3]
    concept_keywords = plan.get("concept_keywords", [])
    concept_paths = plan.get("concepts", [])

    # Read detailed variable definitions
    variable_definitions = []
    
    for path in dict_paths:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    var_name = row.get("Variable", "").strip()
                    if variable and var_name.upper() == variable.upper():
                        variable_definitions.append({
                            "file": Path(path).name,
                            "category": Path(path).parent.name,
                            "variable": var_name,
                            "type": row.get("Type", ""),
                            "length": row.get("Length", ""),
                            "format": row.get("Format", ""),
                            "label": row.get("Label", "")
                        })
                        logger.info(f"Found definition for {var_name} in {path}")
                        break
        except Exception as e:
            logger.warning(f"Failed to read definition from {path}: {e}")
    
    # Read concept file contents
    concept_contents = []
    for path in concept_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Limit length to avoid exceeding token limit
                if len(content) > 10000:
                    content = content[:10000] + "\n... (content truncated due to length)"
                concept_contents.append({
                    "file": Path(path).name,
                    "path": path,
                    "content": content
                })
                logger.info(f"Loaded concept file: {path} ({len(content)} chars)")
        except Exception as e:
            logger.warning(f"Failed to read concept file {path}: {e}")

    # Build prompt (adjust based on query type)
    # Prioritize concept content, then variable definitions
    if concept_contents and concept_keywords:
        # Concept query (must have both file content and keywords)
        system = textwrap.dedent("""
        You are a transplant data analysis assistant. User is asking about medical concepts (e.g., eGFR, KDPI).
        
        You will receive:
        1. User's question
        2. Related R code file content (may include formulas, comments, examples)
        
        Please provide:
        1. Medical definition and clinical significance of the concept
        2. If there's a calculation formula in the code, explain the method and parameters
        3. How to understand and use this concept
        4. If user requests code, extract or summarize relevant R code snippets
        
        Requirements:
        - Answer professionally and clearly
        - Base on provided file content, don't fabricate
        - Make full use of code comments if present
        - Keep well-organized
        """).strip()
        
        user_content = {
            "user_query": user_query,
            "concept_keywords": concept_keywords,
            "concept_files": concept_contents
        }
    elif variable_definitions:
        # Variable definition query (has actual variable definition data)
        system = textwrap.dedent("""
        You are a transplant data dictionary assistant. Your task is to explain SRTR data dictionary variable definitions.
        
        You will receive:
        1. User's question
        2. Variable definitions extracted from data dictionary files (including type, length, format, label)
        
        Please provide:
        1. Variable's meaning and description
        2. Data type and format explanation
        3. Which data table/category this variable is in
        4. How to understand and use this variable (if applicable)
        
        Requirements:
        - Answer professionally and clearly
        - Base on actual provided definitions, don't fabricate
        - Honestly inform if definition not found
        - Keep concise, highlight key points
        """).strip()
        
        user_content = {
            "user_query": user_query,
            "variable": variable,
            "variable_definitions": variable_definitions,
            "dictionary_files": [Path(p).name for p in dict_paths]
        }
    else:
        # General query (no specific content found)
        system = textwrap.dedent("""
        You are a transplant data assistant. User asked about SRTR data but we didn't find related definitions or files.
        
        Please:
        1. Based on the question itself, provide relevant information you know
        2. Explain we didn't find specific definitions in local database
        3. Suggest how user can rephrase the question or provide more information
        
        Requirements:
        - Answer friendly and helpful
        - Be honest about limitations
        - Provide constructive suggestions
        """).strip()
        
        user_content = {
            "user_query": user_query,
            "variable": variable,
            "concept_keywords": concept_keywords,
            "found_paths": dict_paths + concept_paths
        }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False, indent=2)}
    ]

    try:
        content = _openai_chat(messages, temperature=0.2, max_tokens=2000)
    except Exception as e:
        logger.exception("Agent A failed to generate answer")
        
        # Fallback handling
        if concept_contents:
            # Has concept file but API failed, show file summary
            content = f"""Found concept file but API call failed.

File: {concept_contents[0]['file']}
Path: {concept_contents[0]['path']}

Content preview:
{concept_contents[0]['content'][:500]}...

(Cannot generate detailed explanation due to API error: {str(e)})"""
        elif variable_definitions:
            # Has definition but API failed, provide basic info
            defs = variable_definitions[0]
            content = f"""Found variable definition:

Variable: {defs['variable']}
Label: {defs['label']}
Type: {defs['type']}
Length: {defs['length']}
Format: {defs['format']}
File: {defs['file']} ({defs['category']})

(Cannot generate detailed explanation due to API error: {str(e)})"""
        else:
            content = f"Sorry, error occurred while generating answer: {str(e)}"

    return {
        "answer_text": content,
        "code": None,
        "runner": None,
        "inputs": {}
    }

# ==========
# Agent B: Controlled Execution (Placeholder)
# ==========

def _agent_b_execute_code(runner: str, code: str, workdir: str) -> Dict[str, Any]:
    """
    TODO: Execute in restricted sandbox (recommend separate container)
    Currently a placeholder implementation
    """
    job_id = uuid.uuid4().hex[:8]
    job_dir = os.path.join("storage", "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save code
    code_path = os.path.join(job_dir, "snippet.R" if runner == "r" else "snippet.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code or "")

    # Placeholder execution
    fake_out = os.path.join(job_dir, "out.csv")
    with open(fake_out, "w", encoding="utf-8") as f:
        f.write("result,note\n42,agent-b-placeholder\n")

    return {
        "status": "ok",
        "job_id": job_id,
        "artifacts": [
            {"type": "table", "path": fake_out},
            {"type": "code", "path": code_path}
        ],
        "logs": "Agent B placeholder: no real execution yet."
    }

# ==========
# HTTP Endpoints
# ==========

@csrf_exempt
def api_query(request):
    """
    POST /api/query
    Input: { "prompt": "..." }
    Return: paths, answer_text, code, runner, inputs, trace
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    user_prompt = (data or {}).get("prompt", "").strip()
    if not user_prompt:
        return JsonResponse({"error": "No prompt provided"}, status=400)

    try:
        # 1) Initial Agent: Path planning
        plan = _initial_agent_plan_locally(user_prompt)
        logger.info(f"Initial Agent plan: {plan}")

        # 2) Agent A: Generate answer
        a_out = _agent_a_generate_answer(user_prompt, plan)
        logger.info(f"Agent A output generated")

        # Build response - ensure all paths are correctly passed
        resp = {
            "paths": {
                "concepts": plan.get("concepts", []),
                "dictionaries": plan.get("dictionaries", []),
                "data": plan.get("data", []),
                "docs": plan.get("docs", []),
                "intent": plan.get("intent", []),
            },
            "answer_text": a_out.get("answer_text", ""),
            "code": a_out.get("code"),
            "runner": a_out.get("runner"),
            "inputs": a_out.get("inputs", {}),
            "trace": {
                "rationale": plan.get("rationale", ""),
                "variable": plan.get("variable"),
                "concept_keywords": plan.get("concept_keywords", [])
            }
        }
        
        # Debug logging: record complete returned data
        logger.info(f"API Response paths: concepts={len(resp['paths']['concepts'])}, "
                   f"dictionaries={len(resp['paths']['dictionaries'])}, "
                   f"intent={resp['paths']['intent']}")
        
        return JsonResponse(resp)
    
    except Exception as e:
        logger.exception("Error in api_query")
        return JsonResponse({
            "error": str(e),
            "answer_text": f"Error occurred while processing request: {str(e)}",
            "paths": {
                "concepts": [],
                "dictionaries": [],
                "data": [],
                "docs": [],
                "intent": ["error"]
            },
            "trace": {
                "error": str(e)
            }
        }, status=500)

@csrf_exempt
def api_execute(request):
    """
    POST /api/execute
    Input: { "runner": "r"|"python", "code": "...", "workdir": "..." }
    Return: { "status": "ok", "artifacts": [...], "logs": "..." }
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    runner = (data or {}).get("runner")
    code = (data or {}).get("code")
    workdir = (data or {}).get("workdir") or (data or {}).get("inputs", {}).get("workdir")

    if not runner or runner not in ("r", "python"):
        return JsonResponse({"error": "Invalid runner"}, status=400)
    if not code:
        return JsonResponse({"error": "No code provided"}, status=400)
    if not workdir:
        return JsonResponse({"error": "No workdir provided"}, status=400)

    # Security check
    try:
        workdir_path = Path(workdir).resolve()
        concepts_path = CONCEPTS_DIR.resolve()
        if not str(workdir_path).startswith(str(concepts_path)):
            return JsonResponse({"error": "Invalid workdir: must be under concepts/"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Invalid workdir path: {e}"}, status=400)

    try:
        out = _agent_b_execute_code(runner=runner, code=code, workdir=workdir)
        return JsonResponse(out)
    except Exception as e:
        logger.exception("Error in api_execute")
        return JsonResponse({
            "error": str(e),
            "status": "error",
            "logs": f"Execution failed: {str(e)}"
        }, status=500)
    
# ==========
# 管理端点
# ==========

@csrf_exempt
def api_rebuild_index(request):
    """
    POST /api/rebuild_index
    手动重建字典索引
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        index = build_dictionaries_index()
        return JsonResponse({
            "status": "ok",
            "message": "Dictionary index rebuilt successfully",
            "stats": {
                "files": len(index.get("files", [])),
                "variables": len(index.get("variables", {})),
                "total_variables": index.get("total_variables", 0)
            }
        })
    except Exception as e:
        logger.exception("Error rebuilding index")
        return JsonResponse({
            "error": str(e),
            "status": "error"
        }, status=500)

@csrf_exempt
def api_debug(request):
    """
    GET /api/debug?q=CAN_ABO
    调试端点：显示变量提取和路径匹配的详细信息
    """
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    query = request.GET.get("q", "")
    if not query:
        return JsonResponse({"error": "No query provided"}, status=400)
    
    try:
        # 提取变量
        m = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", query)
        var_token = None
        for match in m:
            if "_" in match:
                var_token = match
                break
        if not var_token and m:
            var_token = m[0]
        
        # 加载索引
        idx = load_dictionaries_index()
        
        # 尝试匹配
        matches = guess_paths_for_variable(var_token, idx, topk=5) if var_token else []
        
        # 检查索引中是否有这个变量
        var_upper = var_token.upper() if var_token else ""
        in_index = var_upper in idx.get("variables", {})
        
        return JsonResponse({
            "query": query,
            "all_matches": m,
            "selected_variable": var_token,
            "variable_upper": var_upper,
            "in_index": in_index,
            "index_entry": idx.get("variables", {}).get(var_upper, []),
            "guessed_paths": matches,
            "total_index_variables": len(idx.get("variables", {})),
            "sample_variables": list(idx.get("variables", {}).keys())[:20]
        })
    except Exception as e:
        logger.exception("Error in debug")
        return JsonResponse({
            "error": str(e)
        }, status=500)