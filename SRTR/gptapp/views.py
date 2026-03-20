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
from bs4 import BeautifulSoup

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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def _openai_chat(messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Minimal wrapper for OpenAI Chat Completions API."""
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

def _get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using OpenAI API."""
    url = f"{OPENAI_API_BASE}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error: {resp.status_code}")
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        logger.exception("Embedding request failed: %s", e)
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
HTML_MIRROR_DIR = DATA_REPO / "html_mirrors"  # Store HTML mirror files
CHUNKS_DIR = DATA_REPO / "chunks"  # Store parsed chunks
EMBEDDINGS_DIR = DATA_REPO / "embeddings"  # Store vector embeddings
DICT_ROOT = DATA_REPO / "dictionaries"
CONCEPTS_DIR = DATA_REPO / "concepts"
DOCS_DIR = DATA_REPO / "docs"

# Document metadata index
DOCS_INDEX_PATH = DATA_REPO / "meta" / "documents.index.json"
CHUNKS_INDEX_PATH = DATA_REPO / "meta" / "chunks.index.json"

# dictionary index
DICT_INDEX = DATA_REPO / "meta" / "dictionaries.index.json"


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
# HTML Parser Layer
# ==========

def parse_metric_definition_html(html: str) -> List[Dict[str, Any]]:
    """
    Parse metrics definition HTML page.
    Returns list of metric definitions with structure:
    [
      {
        "metric_code": "1YR_PAT_SURV",
        "title": "1-year patient survival",
        "definition": "Definition text...",
        "notes": "Additional notes..."
      },
      ...
    ]
    """
    soup = BeautifulSoup(html, 'html.parser')
    metrics = []
    
    # Example parsing logic - adjust based on actual HTML structure
    metric_sections = soup.find_all('div', class_='metric-definition')
    
    for section in metric_sections:
        try:
            metric = {
                "metric_code": section.get('data-metric-code', ''),
                "title": section.find('h3').text.strip() if section.find('h3') else '',
                "definition": section.find('div', class_='definition').text.strip() if section.find('div', class_='definition') else '',
                "notes": section.find('div', class_='notes').text.strip() if section.find('div', class_='notes') else ''
            }
            metrics.append(metric)
        except Exception as e:
            logger.warning(f"Failed to parse metric section: {e}")
            continue
    
    return metrics

def parse_wait_time_html(html: str) -> Dict[str, Any]:
    """
    Parse wait time calculator/methodology page.
    Returns structured data about the wait time model.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    result = {
        "title": "",
        "overview": "",
        "input_variables": [],
        "methodology": "",
        "interpretation": ""
    }
    
    # Adjust based on actual HTML structure
    if soup.find('h1'):
        result["title"] = soup.find('h1').text.strip()
    
    overview_section = soup.find('div', class_='overview')
    if overview_section:
        result["overview"] = overview_section.text.strip()
    
    # Parse input variables table
    variables_table = soup.find('table', class_='input-variables')
    if variables_table:
        for row in variables_table.find_all('tr')[1:]:  # Skip header
            cols = row.find_all('td')
            if len(cols) >= 2:
                result["input_variables"].append({
                    "name": cols[0].text.strip(),
                    "description": cols[1].text.strip()
                })
    
    return result

def parse_center_html(html: str) -> Dict[str, Any]:
    """
    Parse transplant center page.
    Returns center information and key metrics.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    result = {
        "center_name": "",
        "center_id": "",
        "location": "",
        "organ_programs": [],
        "metrics": []
    }
    
    # Adjust based on actual HTML structure
    if soup.find('h1', class_='center-name'):
        result["center_name"] = soup.find('h1', class_='center-name').text.strip()
    
    # Parse metrics table
    metrics_table = soup.find('table', class_='center-metrics')
    if metrics_table:
        for row in metrics_table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                result["metrics"].append({
                    "metric": cols[0].text.strip(),
                    "value": cols[1].text.strip(),
                    "national_avg": cols[2].text.strip()
                })
    
    return result

def parse_generic_html(html: str) -> Dict[str, Any]:
    """
    Generic HTML parser for pages without specific structure.
    Extracts title, main content sections, and links.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer"]):
        script.decompose()
    
    result = {
        "title": soup.find('title').text.strip() if soup.find('title') else "",
        "h1": soup.find('h1').text.strip() if soup.find('h1') else "",
        "sections": []
    }
    
    # Extract main content sections
    main_content = soup.find('main') or soup.find('div', class_='content') or soup.body
    if main_content:
        for section in main_content.find_all(['section', 'div'], class_=re.compile(r'section|content-block')):
            section_title = section.find(['h2', 'h3'])
            section_text = section.get_text(strip=True, separator=' ')
            
            if section_text:
                result["sections"].append({
                    "title": section_title.text.strip() if section_title else "",
                    "content": section_text[:1000]  # Limit length
                })
    
    return result

def load_and_parse_html(doc_id: str, doc_type: str) -> Dict[str, Any]:
    """
    Load HTML file and parse based on document type.
    
    Args:
        doc_id: Document identifier
        doc_type: Type of document (metric_definition, wait_time, center_page, etc.)
    
    Returns:
        Parsed structured data
    """
    # Load document metadata
    docs_index = load_documents_index()
    if doc_id not in docs_index:
        raise ValueError(f"Document {doc_id} not found in index")
    
    doc_meta = docs_index[doc_id]
    html_path = HTML_MIRROR_DIR / doc_meta["local_path"]
    
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse based on document type
    if doc_type == "metric_definition":
        return parse_metric_definition_html(html_content)
    elif doc_type == "wait_time":
        return parse_wait_time_html(html_content)
    elif doc_type == "center_page":
        return parse_center_html(html_content)
    else:
        return parse_generic_html(html_content)

# ==========
# Document and Chunk Index Management
# ==========

def load_documents_index() -> Dict[str, Dict[str, Any]]:
    """
    Load document metadata index.
    
    Format:
    {
      "metrics_definitions": {
        "doc_id": "metrics_definitions",
        "url": "https://www.srtr.org/...",
        "local_path": "metrics_definitions.html",
        "doc_type": "metric_definition",
        "organ": null,
        "last_updated": "2025-01-01"
      },
      ...
    }
    """
    if not DOCS_INDEX_PATH.exists():
        logger.warning("Documents index not found, returning empty index")
        return {}
    
    with open(DOCS_INDEX_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_chunks_index() -> Dict[str, Dict[str, Any]]:
    """
    Load chunks metadata index.
    
    Format:
    {
      "ch_001": {
        "chunk_id": "ch_001",
        "doc_id": "metrics_definitions",
        "text": "1-year patient survival is defined as...",
        "section_title": "1-year patient survival",
        "metadata": {
          "metric_code": "1YR_PAT_SURV",
          "doc_type": "metric_definition"
        }
      },
      ...
    }
    """
    if not CHUNKS_INDEX_PATH.exists():
        logger.warning("Chunks index not found, returning empty index")
        return {}
    
    with open(CHUNKS_INDEX_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_chunks_index(chunks_index: Dict[str, Dict[str, Any]]):
    """Save chunks index to file."""
    CHUNKS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_index, f, ensure_ascii=False, indent=2)

# ==========
# RAG Retrieval Pipeline
# ==========

def retrieve_chunks(query: str, filters: Optional[Dict[str, Any]] = None, 
                   semantic_scope: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant text chunks based on semantic similarity.
    
    Enhanced with two-stage filtering:
    1. Pre-filter by semantic_scope (keyword matching)
    2. Semantic similarity search on filtered subset
    
    Args:
        query: User's question
        filters: Metadata filters (doc_type, metric_code, center_id, etc.)
        semantic_scope: List of key terms for pre-filtering (e.g., ["patient survival", "1-year outcomes"])
        top_k: Number of chunks to return
    
    Returns:
        List of chunks with metadata, sorted by similarity
    """
    try:
        # Load chunks index
        chunks_index = load_chunks_index()
        
        if not chunks_index:
            logger.warning("No chunks available for retrieval")
            return []
        
        # Stage 1: Pre-filter by semantic scope and metadata
        candidate_chunks = []
        
        for chunk_id, chunk in chunks_index.items():
            # Apply metadata filters
            if filters and not _matches_filters(chunk, filters):
                continue
            
            # Apply semantic scope filtering (keyword-based)
            if semantic_scope and not _matches_semantic_scope(chunk, semantic_scope):
                continue
            
            candidate_chunks.append((chunk_id, chunk))
        
        if not candidate_chunks:
            logger.warning(f"No chunks passed pre-filtering. Filters: {filters}, Scope: {semantic_scope}")
            return []
        
        logger.info(f"Pre-filtering: {len(candidate_chunks)} candidates from {len(chunks_index)} total chunks")
        
        # Stage 2: Semantic similarity search
        query_embedding = _get_embedding(query)
        scored_chunks = []
        
        for chunk_id, chunk in candidate_chunks:
            # Load chunk embedding
            embedding_path = EMBEDDINGS_DIR / f"{chunk_id}.json"
            if not embedding_path.exists():
                continue
            
            with open(embedding_path, 'r') as f:
                chunk_embedding = json.load(f)["embedding"]
            
            # Calculate cosine similarity
            similarity = _cosine_similarity(query_embedding, chunk_embedding)
            
            scored_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "section_title": chunk.get("section_title", ""),
                "doc_type": chunk["metadata"].get("doc_type", ""),
                "source_url": chunk["metadata"].get("source_url", ""),
                "similarity": similarity,
                "metadata": chunk["metadata"]
            })
        
        # Sort by similarity and return top_k
        scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        results = scored_chunks[:top_k]
        
        if results:
            logger.info(f"Retrieved {len(results)} chunks. Top similarity: {results[0]['similarity']:.3f}")
        else:
            logger.warning("No chunks retrieved after similarity calculation")
        
        return results
        
    except Exception as e:
        logger.exception(f"Error in retrieve_chunks: {e}")
        return []

def _matches_semantic_scope(chunk: Dict[str, Any], semantic_scope: List[str]) -> bool:
    """
    Check if chunk matches semantic scope using keyword matching.
    
    Looks in:
    - chunk text
    - section_title
    - metadata.semantic_scope (if exists)
    """
    # Get searchable text
    searchable = []
    searchable.append(chunk.get("text", "").lower())
    searchable.append(chunk.get("section_title", "").lower())
    
    # Check if chunk has its own semantic_scope metadata
    chunk_scope = chunk.get("metadata", {}).get("semantic_scope", [])
    if chunk_scope:
        searchable.extend([s.lower() for s in chunk_scope])
    
    combined_text = " ".join(searchable)
    
    # Check if any scope term appears in the chunk
    matches = 0
    for scope_term in semantic_scope:
        if scope_term.lower() in combined_text:
            matches += 1
    
    # Require at least one match
    return matches > 0

def _matches_filters(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if chunk matches the given filters."""
    metadata = chunk.get("metadata", {})
    
    for key, value in filters.items():
        if key in metadata:
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        else:
            # Check if filter key exists in chunk itself
            if key in chunk and chunk[key] != value:
                return False
    
    return True

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def _extract_variable_definitions(var_name: str, paths: List[str]) -> List[Dict[str, Any]]:
    """Extract variable definition from one or more dictionary CSV files."""
    definitions = []
    var_upper = var_name.strip().upper()

    for csv_path_str in paths:
        csv_path = Path(csv_path_str)
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
            
            if content is None: continue

            reader = csv.DictReader(StringIO(content))
            
            for row in reader:
                if row.get("Variable", "").strip().upper() == var_upper:
                    definitions.append({
                        "variable": var_upper,
                        "file": csv_path.name,
                        "category": csv_path.parent.name,
                        "path": csv_path_str,
                        "type": row.get("Type", ""),
                        "length": row.get("Length", ""),
                        "format": row.get("Format", ""),
                        "label": row.get("Label", "")
                    })
                    break
        except Exception as e:
            logger.error(f"Failed to read/parse {csv_path_str} for {var_name}: {e}")
            continue
    return definitions

def _extract_concept_contents(concept_keywords: List[str], paths: List[str]) -> Dict[str, str]:
    """Extract concept content (e.g., R code/comments) from files."""
    concept_contents = {}
    for p in paths:
        path_obj = Path(p)
        try:
            # For R files, primarily read the file content
            if path_obj.suffix.lower() == ".r":
                with open(path_obj, "r", encoding="utf-8") as f:
                    concept_contents[path_obj.name] = f.read()
        except Exception as e:
            logger.error(f"Failed to read concept file {p}: {e}")
            continue
    return concept_contents

def _generate_fallback_answer(reason: str, e: Optional[Exception] = None) -> Dict[str, Any]:
    """
    Handle various error situations and unsuccessful retrieval cases.
    Since it is not defined in views.py, we create a basic version.
    """
    logger.error(f"Fallback triggered for reason: {reason}. Exception: {e}" if e else f"Fallback triggered for reason: {reason}.")
    return {
        "summary": f"Sorry, the system is unable to answer your question. Reason: {reason}.",
        "detail": "Please try phrasing your question differently or consult another data dictionary or concept.",
        "sources": [],
        "confidence": "low",
        "process_info": {
            "type": "Fallback/Error Handling",
            "paths": [f"Internal/Fallback_Code:{reason}"]
        },
        "error": True
    }

def _generate_answer_from_definition(query, definitions):
    # Simulated LLM call to generate summary and detailed information
    if definitions:
        d = definitions[0]
        summary = f"Found the definition for variable **{d['variable']}**."
        detail = f"""
        **Variable Name:** {d['variable']}
        **Description (Label):** {d['label'] or 'N/A'}
        **Type:** {d['type']}, **Length:** {d['length']}, **Format:** {d['format']}
        **Source File:** {d['file']} ({d['category']})
        """
        return {"summary": summary, "detail": detail}
    return {"summary": "No related definition found", "detail": "Please check whether the variable name is correct."}

def _generate_answer_from_concept(query, keywords, contents):
    # Simulated LLM call to generate summary and detailed information
    if contents:
        key = list(contents.keys())[0]
        content_snippet = contents[key][:500] + "..."
        summary = f"Successfully extracted the explanation for concept(s): **{', '.join(keywords)}**."
        detail = f"Extracted relevant code and annotations from file **{key}**. File content summary:\n\n```R\n{content_snippet}\n```"
        return {"summary": summary, "detail": detail}
    return {"summary": "No related concept file found", "detail": "Please check whether the concept file exists."}

# views.py (修改 _handle_data_dictionary_query 函数)
# 确保 _openai_chat 在 views.py 中可用


def _handle_data_dictionary_query(query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle data dictionary variable queries, implementing the file reading and
    LLM generation logic from views_local, with robust variable extraction.
    """
    
    # --- Step 1: 提取变量名 (优先从 Initial Agent 的计划中获取) ---
    # Initial Agent 返回的 key 可能是 'variable_name'
    var_token = filters.get("variable_name") or filters.get("variable")
    
    if not var_token:
        # 如果 Initial Agent 没有成功提取，则使用 views_local 中的正则表达式逻辑重新提取
        m = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", query)
        var_token = next((match for match in m if "_" in match), m[0] if m else None)
    
    if not var_token:
        return _generate_fallback_answer("no_variable_found")

    try:
        # 依赖 load_dictionaries_index 和 guess_paths_for_variable (假设已迁移)
        idx = load_dictionaries_index()
        dict_paths = guess_paths_for_variable(var_token, idx, topk=3)
        
        if not dict_paths:
             logger.warning(f"Variable '{var_token}' found in query but no paths were guessed.")
             return _generate_fallback_answer(f"Variable '{var_token}' not indexed.")

        # --- Step 2: 实现 views_local 的文件读取逻辑 (确保查找逻辑无误) ---
        variable_definitions = []
        for path_str in dict_paths:
            path = Path(path_str)
            try:
                # 使用 utf-8-sig 应对可能的 BOM (这是最关键的一步)
                with open(path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 确保列名是 'Variable' 且进行大小写不敏感匹配
                        var_name = row.get("Variable", "").strip() 
                        if var_name.upper() == var_token.upper(): 
                            variable_definitions.append({
                                "file": path.name,
                                "category": path.parent.name,
                                "variable": var_name,
                                "type": row.get("Type", ""),
                                "length": row.get("Length", ""),
                                "format": row.get("Format", ""),
                                "label": row.get("Label", ""),
                                "path": path_str,
                            })
                            logger.info(f"Found definition for {var_name} in {path}")
                            break 
            except Exception as e:
                logger.warning(f"Failed to read definition from {path}: {e}")
        
        if not variable_definitions:
             # 这仍然是失败的最高可能性：找到了文件路径，但文件中没有匹配的行
             return _generate_fallback_answer(f"Variable '{var_token}' found paths but no row matched in CSV.")

        # --- Step 3: 构建提示词 (与 views_local 保持一致) ---
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
        
        user_content_dict = {
            "user_query": query,
            "variable": var_token,
            "variable_definitions": variable_definitions,
            "dictionary_files": [Path(d['path']).name for d in variable_definitions]
        }
        user_content = json.dumps(user_content_dict, ensure_ascii=False, indent=2)
        print("load info...")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]
        
        # --- Step 4: 调用 LLM 并格式化结果 ---
        llm_response_text = _openai_chat(messages, temperature=0.0)
        
        summary = llm_response_text.split('\n')[0] if llm_response_text else f"找到了变量 {var_token} 的定义。"
        detail = llm_response_text
        paths_used = [d['path'] for d in variable_definitions]
        
        return {
            "summary": summary,
            "detail": detail,
            "sources": [],
            "confidence": "high",
            "process_info": {
                "type": "Data Dictionary Lookup (本地字典)",
                "paths": paths_used
            }
        }
        
    except Exception as e:
        logger.exception(f"Data dictionary query failed: {e}")
        return _generate_fallback_answer("data_dictionary_query_error", e=e)
# views.py (修改 _handle_concept_query 函数)
# 确保 _openai_chat 在 views.py 中可用

def _handle_concept_query(query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """处理 SRTR 概念查询，使用本地 R 代码和 LLM 生成解释。"""
    # 提取概念文件和关键词的逻辑保持不变...
    concept_paths = []
    concept_r = CONCEPTS_DIR / "concepts.R" # 假设 concepts.R 存在
    if concept_r.exists():
        concept_paths.append(str(concept_r))
    
    concept_keywords = plan.get("entity_identifiers", {}).get("concept_keywords", [])
    concept_contents = _extract_concept_contents(concept_keywords, concept_paths)

    if not concept_contents:
        return _generate_fallback_answer("no_concept_content_found")

    # === 整合：概念查询提示词逻辑 ===
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
    
    user_content_dict = {
        "user_query": query,
        "concept_keywords": concept_keywords,
        "concept_files": concept_contents
    }
    user_content = json.dumps(user_content_dict, ensure_ascii=False, indent=2)

    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]
        
        # 调用 _openai_chat 
        llm_response_text = _openai_chat(messages, temperature=0.2)
        
        # 假设 LLM 返回纯文本，需要自己分割成 summary 和 detail
        summary = llm_response_text.split('\n')[0] if llm_response_text else "概念解释生成成功。"
        detail = llm_response_text
        paths_used = list(concept_contents.keys())
        
        return {
            "summary": summary,
            "detail": detail,
            "sources": [],
            "confidence": "high",
            "process_info": {
                "type": "Concept/Model Explanation (本地代码)",
                "paths": paths_used
            }
        }
    except Exception as e:
        return _generate_fallback_answer("concept_query_llm_error", e=e)


# ==========
# Initial Agent: Intent Classification and Path Planning
# ==========

def _initial_agent_plan(user_query: str) -> dict:
    """
    Initial Agent: Classify intent and determine retrieval strategy.
    
    Enhanced to return:
    - Specific entity identifiers (metric_code, center_id, etc.)
    - Answer mode (numeric vs textual)
    - Semantic scope for filtering
    
    Returns:
    {
      "intent": "metric_definition" | "center_comparison" | "wait_time" | ...,
      "answer_mode": "numeric" | "textual",
      "entity_identifiers": {
        "metric_code": "1YR_PAT_SURV",
        "center_id": "UCLA_kidney",
        "organ": "kidney"
      },
      "semantic_scope": ["patient survival", "1-year outcomes"],
      "filters": {...},
      "retrieval_needed": true/false,
      "use_deterministic_tool": true/false,
      "rationale": "..."
    }
    """
    system = textwrap.dedent("""
    You are the Initial Agent for SRTR data query system.  
    Your task: Analyze the user's question and determine:
    1. Intent classification
    2. **Answer mode**: Is this a numeric query or textual query?
    3. **Tool detection**: Is this a calculator/tool request?
    4. **Entity identifiers**: Extract specific codes, IDs, or names mentioned
    5. **Semantic scope**: Key concepts/terms that define the query scope
    6. Appropriate filters and retrieval strategy
        
    Intent categories:
    - "calculator": User wants to calculate something (waiting time, KDPI, EPTS, etc.)
    - "metric_definition": User asks what a metric means, how it's calculated
    - "center_comparison": User compares centers or wants specific center data
    - "wait_time_explanation": User asks about wait time methodology (different from calculator!)
    - "methodology": User asks about SRTR methodology
    - "opo_info": User asks about OPOs
    - "data_dictionary_lookup": **[NEW] User asks about specific variable definitions (e.g., TX_DATE)**
    - "concept_explanation": **[NEW] User asks for explanation of a key concept or model (e.g., KDPI, eGFR)**
    - "general": General transplantation question
        
    Answer modes:
    - "calculator": User wants to run a calculation with their own data
    - "numeric": User wants specific numbers/statistics from database
    - "textual": User wants explanations/definitions
        
    Tool names (for calculator intent):
    - "kidney_waiting_time": Kidney transplant waiting time calculator
    - "kdpi": Kidney Donor Profile Index calculator
    - "epts": Estimated Post-Transplant Survival calculator
        
    Entity identifiers (extract if present):
    - tool_name: Which calculator/tool (if intent is "calculator")
    - metric_code: Standard SRTR metric codes
    - center_id: Transplant center identifier
    - organ: Organ type
    - **variable_name**: **[NEW] The exact, capitalized variable name (e.g., 'TX_DATE')**
    - **concept_keywords**: **[NEW] Key concept acronyms or names (e.g., ['KDPI', 'eGFR'])**
        
    Semantic scope: List 2-5 key terms that define what this query is about.
        
    Return ONLY JSON format, without any annotationS:
    {
    "intent": "...",
    "answer_mode": "calculator" | "numeric" | "textual",
    "entity_identifiers": {
        "tool_name": "...",
        "metric_code": "...",
        "center_id": "...",
        "organ": "...",
        "variable_name": "...",
        "concept_keywords": [...]
    },
    "semantic_scope": [...],
    "filters": {...},
    "retrieval_needed": true/false,
    "use_deterministic_tool": true/false,
    "rationale": "brief explanation"
    }
        
    CRITICAL: Detect calculator requests!
    Keywords that indicate calculator intent:
    - "calculate my waiting time"
    - "how long will I wait"
    - "I'm [age], type [blood_type], dialysis [time]" (giving personal data)
    - "estimate my wait"
    - "calculate KDPI"
    - "what is my EPTS score"
    
    Examples:
                             
    Q: "What is the definition of TX_DATE?"
    A: {
    "intent": "data_dictionary_lookup",
    "answer_mode": "textual",
    "entity_identifiers": {"variable_name": "TX_DATE"},
    "semantic_scope": ["TX_DATE", "data dictionary"],
    "filters": {},
    "retrieval_needed": false,
    "use_deterministic_tool": true,
    "rationale": "User is asking for a specific, capitalized variable definition, requiring a local dictionary lookup."
    }

    Q: "How is eGFR calculated and what variables are needed?"
    A: {
    "intent": "concept_explanation",
    "answer_mode": "textual",
    "entity_identifiers": {"concept_keywords": ["eGFR"]},
    "semantic_scope": ["eGFR", "kidney function", "calculation methodology"],
    "filters": {},
    "retrieval_needed": false,
    "use_deterministic_tool": true,
    "rationale": "User is asking for an explanation of a key concept (eGFR), requiring a local concept file lookup. The query does not require RAG."
    }

    Q: "How is kidney waiting time calculated?"
    A: {
    "intent": "wait_time_explanation",
    "answer_mode": "textual",
    "entity_identifiers": {"organ": "kidney"},
    "semantic_scope": ["waiting time", "calculation methodology"],
    "filters": {"doc_type": "wait_time"},
    "retrieval_needed": true,
    "use_deterministic_tool": false,
    "rationale": "User wants explanation of methodology, not personal calculation"
    }
        
    Q: "I'm 45 years old, type O, been on dialysis for 2 years, CPRA is 85. How long will I wait?"
    A: {
      "intent": "calculator",
      "answer_mode": "calculator",
      "entity_identifiers": {"tool_name": "kidney_waiting_time", "organ": "kidney"},
      "semantic_scope": ["waiting time", "kidney transplant"],
      "filters": {},
      "retrieval_needed": false,
      "use_deterministic_tool": true,
      "rationale": "User provided personal data and asks for wait time calculation"
    }
    
    Q: "Calculate my kidney waiting time. I'm 30, AB blood type, 6 months on dialysis"
    A: {
      "intent": "calculator",
      "answer_mode": "calculator",
      "entity_identifiers": {"tool_name": "kidney_waiting_time", "organ": "kidney"},
      "semantic_scope": ["waiting time", "kidney transplant"],
      "filters": {},
      "retrieval_needed": false,
      "use_deterministic_tool": true,
      "rationale": "Explicit calculator request with parameters"
    }
    
    
    Q: "What is 1-year patient survival?"
    A: {
      "intent": "metric_definition",
      "answer_mode": "textual",
      "entity_identifiers": {"metric_code": "1YR_PAT_SURV"},
      "semantic_scope": ["patient survival", "1-year outcomes"],
      "filters": {"doc_type": "metric_definition", "metric_code": "1YR_PAT_SURV"},
      "retrieval_needed": true,
      "use_deterministic_tool": false,
      "rationale": "User wants definition of specific metric"
    }
    
    Q: "What is UCLA's 1-year kidney survival rate?"
    A: {
      "intent": "center_comparison",
      "answer_mode": "numeric",
      "entity_identifiers": {"center_id": "UCLA_kidney", "metric_code": "1YR_GRF_SURV", "organ": "kidney"},
      "semantic_scope": ["graft survival", "UCLA outcomes"],
      "filters": {"center_id": "UCLA_kidney", "organ": "kidney"},
      "retrieval_needed": false,
      "use_deterministic_tool": true,
      "rationale": "User wants specific numeric data from database"
    }
    """).strip()
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query}
    ]
    
    try:
        raw_response = _openai_chat(messages, temperature=0.2, max_tokens=600)
        print(raw_response)
        plan = json.loads(raw_response)
        
        # Ensure required fields exist
        plan.setdefault("answer_mode", "textual")
        plan.setdefault("entity_identifiers", {})
        plan.setdefault("semantic_scope", [])
        
        logger.info(f"Initial Agent plan: intent={plan.get('intent')}, "
                   f"answer_mode={plan.get('answer_mode')}, "
                   f"entities={plan.get('entity_identifiers')}")
        return plan
        
    except Exception as e:
        logger.exception(f"Initial Agent failed: {e}")
        # Fallback
        return {
            "intent": "general",
            "answer_mode": "textual",
            "entity_identifiers": {},
            "semantic_scope": [],
            "filters": {},
            "retrieval_needed": True,
            "use_deterministic_tool": False,
            "rationale": f"Fallback due to error: {e}"
        }

# ==========
# Kidney Waiting Time Calculator (Demo Tool)
# ==========

def calculate_kidney_waiting_time(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate predicted kidney transplant waiting time.
    
    This is a simplified demo implementation based on SRTR's calculator.
    Real implementation would use SRTR's actual statistical models.
    
    Parameters:
    - blood_type: "O", "A", "B", "AB"
    - age: int (years)
    - dialysis_time: int (months on dialysis)
    - cpra: int (0-100, calculated Panel Reactive Antibody percentage)
    - diabetes: bool
    - region: int (1-11, UNOS region)
    
    Returns:
    {
      "median_wait_days": int,
      "wait_ranges": {
        "25th_percentile": int,
        "75th_percentile": int
      },
      "factors_impact": {...},
      "calculation_notes": str
    }
    """
    
    # Validate required parameters
    required = ["blood_type", "age", "dialysis_time", "cpra"]
    missing = [p for p in required if p not in params]
    if missing:
        return {
            "error": f"Missing required parameters: {', '.join(missing)}",
            "required_parameters": {
                "blood_type": "O, A, B, or AB",
                "age": "Patient age in years",
                "dialysis_time": "Months on dialysis",
                "cpra": "CPRA percentage (0-100)"
            }
        }
    
    # Extract and validate parameters
    try:
        blood_type = params["blood_type"].upper()
        age = int(params["age"])
        dialysis_time = int(params["dialysis_time"])
        cpra = int(params["cpra"])
        diabetes = params.get("diabetes", False)
        region = params.get("region", 5)  # Default to region 5
        
        if blood_type not in ["O", "A", "B", "AB"]:
            raise ValueError("Invalid blood type")
        if not (0 <= age <= 120):
            raise ValueError("Age must be between 0-120")
        if not (0 <= cpra <= 100):
            raise ValueError("CPRA must be between 0-100")
        if not (1 <= region <= 11):
            raise ValueError("Region must be between 1-11")
            
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid parameter value: {str(e)}"}
    
    # Base waiting time by blood type (in days)
    # Based on simplified SRTR data - O type waits longest
    base_wait = {
        "O": 1825,   # ~5 years
        "A": 1095,   # ~3 years
        "B": 1460,   # ~4 years
        "AB": 730    # ~2 years
    }
    
    median_wait = base_wait[blood_type]
    
    # Adjustment factors
    factors_impact = {}
    
    # Age factor: Younger patients typically wait longer (more competition)
    if age < 18:
        age_multiplier = 1.3
        factors_impact["age"] = {
            "category": "Pediatric (<18)",
            "impact": "+30% wait time",
            "reason": "Smaller donor pool, prioritized for pediatric donors"
        }
    elif age < 50:
        age_multiplier = 1.0
        factors_impact["age"] = {
            "category": "Adult (18-49)",
            "impact": "Baseline",
            "reason": "Standard allocation priority"
        }
    else:
        age_multiplier = 0.85
        factors_impact["age"] = {
            "category": "Senior (50+)",
            "impact": "-15% wait time",
            "reason": "May accept older/ECD donors"
        }
    
    median_wait *= age_multiplier
    
    # CPRA factor: Higher CPRA = harder to match = longer wait
    if cpra >= 98:
        cpra_multiplier = 2.5
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Highly sensitized (98-100%)",
            "impact": "+150% wait time",
            "reason": "Very difficult to find compatible donor"
        }
    elif cpra >= 80:
        cpra_multiplier = 1.8
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Sensitized (80-97%)",
            "impact": "+80% wait time",
            "reason": "Limited compatible donors"
        }
    elif cpra >= 20:
        cpra_multiplier = 1.2
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Moderately sensitized (20-79%)",
            "impact": "+20% wait time",
            "reason": "Some limitations in donor matching"
        }
    else:
        cpra_multiplier = 1.0
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Not sensitized (<20%)",
            "impact": "Baseline",
            "reason": "Broad donor compatibility"
        }
    
    median_wait *= cpra_multiplier
    
    # Dialysis time factor: Waiting time credit
    if dialysis_time >= 36:
        dialysis_benefit = 0.9  # 10% reduction for long waiters
        factors_impact["dialysis_time"] = {
            "value": f"{dialysis_time} months",
            "impact": "-10% wait time",
            "reason": "Waiting time priority (36+ months)"
        }
    else:
        dialysis_benefit = 1.0
        factors_impact["dialysis_time"] = {
            "value": f"{dialysis_time} months",
            "impact": "No adjustment",
            "reason": "Less than 36 months"
        }
    
    median_wait *= dialysis_benefit
    
    # Diabetes factor: Slight increase due to medical complexity
    if diabetes:
        median_wait *= 1.1
        factors_impact["diabetes"] = {
            "value": "Yes",
            "impact": "+10% wait time",
            "reason": "May require more careful donor matching"
        }
    
    # Regional variation (simplified)
    regional_multipliers = {
        1: 1.1,   # Region 1 (Northeast) - longer waits
        2: 1.05,
        3: 0.95,
        4: 0.9,
        5: 1.0,   # Baseline
        6: 0.95,
        7: 0.9,
        8: 1.0,
        9: 1.05,
        10: 0.95,
        11: 1.1
    }
    
    regional_mult = regional_multipliers.get(region, 1.0)
    median_wait *= regional_mult
    
    if regional_mult != 1.0:
        factors_impact["region"] = {
            "value": f"Region {region}",
            "impact": f"{'+' if regional_mult > 1 else ''}{int((regional_mult - 1) * 100)}% wait time",
            "reason": "Regional supply/demand variation"
        }
    
    # Calculate ranges (25th and 75th percentiles)
    # Simplified: ±40% from median
    percentile_25 = int(median_wait * 0.6)
    percentile_75 = int(median_wait * 1.4)
    median_wait = int(median_wait)
    
    # Convert to human-readable format
    def days_to_readable(days):
        years = days // 365
        remaining_days = days % 365
        months = remaining_days // 30
        
        if years > 0:
            if months > 0:
                return f"{years} year{'s' if years > 1 else ''}, {months} month{'s' if months > 1 else ''}"
            return f"{years} year{'s' if years > 1 else ''}"
        elif months > 0:
            return f"{months} month{'s' if months > 1 else ''}"
        else:
            return f"{days} days"
    
    return {
        "median_wait_days": median_wait,
        "median_wait_readable": days_to_readable(median_wait),
        "wait_ranges": {
            "25th_percentile_days": percentile_25,
            "25th_percentile_readable": days_to_readable(percentile_25),
            "median_days": median_wait,
            "median_readable": days_to_readable(median_wait),
            "75th_percentile_days": percentile_75,
            "75th_percentile_readable": days_to_readable(percentile_75),
        },
        "patient_profile": {
            "blood_type": blood_type,
            "age": age,
            "dialysis_time_months": dialysis_time,
            "cpra": cpra,
            "diabetes": diabetes,
            "region": region
        },
        "factors_impact": factors_impact,
        "interpretation": generate_wait_time_interpretation(median_wait, factors_impact),
        "calculation_notes": (
            "This is a simplified demonstration model based on SRTR data trends. "
            "Actual waiting times vary significantly based on many factors. "
            "For official estimates, please use SRTR's online calculator at https://www.srtr.org/tools/"
        ),
        "disclaimer": "This calculation is for educational purposes only and should not be used for medical decision-making."
    }

def generate_wait_time_interpretation(median_days: int, factors: Dict) -> str:
    """Generate human-readable interpretation of waiting time calculation."""
    
    years = median_days / 365
    
    if years < 1:
        timeline = "less than a year"
    elif years < 2:
        timeline = "about 1-2 years"
    elif years < 3:
        timeline = "about 2-3 years"
    elif years < 5:
        timeline = "about 3-5 years"
    else:
        timeline = "5 years or longer"
    
    # Identify key factors
    key_factors = []
    for factor_name, factor_data in factors.items():
        impact = factor_data.get("impact", "")
        if "+" in impact and factor_name != "age":  # Factors that increase wait
            key_factors.append(factor_data.get("reason", ""))
    
    interpretation = f"Based on the provided information, the estimated median waiting time is **{timeline}**."
    
    if key_factors:
        interpretation += f" The main factors extending this wait time are: {'; '.join(key_factors[:3])}."
    
    interpretation += " Remember that 25% of similar patients receive transplants sooner, and 25% wait longer than this median estimate."
    
    return interpretation

def _agent_b_generate_answer(user_query: str, plan: dict) -> dict:
    """
    Agent B: Generate answer based on retrieved chunks or structured data.
    
    Enhanced output structure:
    - summary: Brief one-sentence answer
    - detail: Comprehensive explanation
    - sources: Full source attribution
    - confidence: Quality indicator
    
    Key principle: ONLY answer based on provided chunks. Never fabricate.
    """
    intent = plan.get("intent", "general-query")
    answer_mode = plan.get("answer_mode", "textual")
    use_deterministic_tool = plan.get("use_deterministic_tool", False)
    retrieval_needed = plan.get("retrieval_needed", True)
    filters = plan.get("filters", {})

    # Case 1: Data dictionary variable lookup (Deterministic/Local)
    if intent == "data_dictionary_lookup":
        return _handle_data_dictionary_query(user_query, filters)
    
    # Case 2: SRTR Concept Explanation (Deterministic/Local)
    # 假设 _initial_agent_plan 可以识别出 intent == "concept"
    if intent == "concept":
        return _handle_concept_query(user_query, plan)
    
    # Route based on answer mode
    if answer_mode == "calculator":
        # Calculator queries - extract params and run calculation
        result = _handle_calculator_query(user_query, plan)
        
        # Requirement 1: 添加处理信息
        if "error" not in result:
            result["process_info"] = {
                "type": "Built-in Calculator",
                # 示例：显示调用了哪个内部工具
                "paths": ["Internal/calculate_kidney_waiting_time"] 
            }
        return result
    
    elif answer_mode == "numeric":
        # Numeric queries should use deterministic tools, not RAG
        if use_deterministic_tool:
            result = _handle_numeric_query(user_query, plan)
            # Requirement 1: 添加处理信息
            if "error" not in result:
                result["process_info"] = {
                    "type": "Numeric Data Lookup",
                    "paths": ["Internal/Database_Access"]
                }
            return result
        else:
            return {
                "summary": "This query requires numeric data that should be looked up directly from databases.",
                "detail": "For accurate numeric values (like specific center statistics), we recommend using SRTR's official data tables rather than text-based retrieval.",
                "sources": [],
                "confidence": "low",
                "recommendation": "use_deterministic_tool"
            }
    
    # Textual queries continue with RAG pipeline
    
    # Case 2: RAG retrieval for explanatory content
    if retrieval_needed:
        semantic_scope = plan.get("semantic_scope", [])
        chunks = retrieve_chunks(user_query, filters=filters, semantic_scope=semantic_scope, top_k=5)
        
        # ⑤ Retrieval failure fallback with strict thresholds
        if not chunks:
            return _safe_fail_message("no_chunks_retrieved", user_query)
        
        max_similarity = max(chunk["similarity"] for chunk in chunks)
        if max_similarity < 0.55:
            return _safe_fail_message("low_similarity", user_query, max_sim=max_similarity)
        
        # Filter out low-quality chunks
        high_quality_chunks = [c for c in chunks if c["similarity"] >= 0.55]
        
        if len(high_quality_chunks) < 2:
            return _safe_fail_message("insufficient_quality_chunks", user_query, 
                                     count=len(high_quality_chunks))
        
        result = _generate_answer_from_chunks(user_query, high_quality_chunks)
        
        # Requirement 1: 添加处理信息
        if "error" not in result:
             result["process_info"] = {
                "type": "RAG Retrieval (Documents)",
                "paths": [c.get("url", c.get("doc_type")) for c in chunks]
            }
        return result
    
    # Case 3: No retrieval needed (shouldn't happen often)
    return {
        "summary": "Unable to determine appropriate retrieval strategy.",
        "detail": "Please rephrase your question or provide more specific details.",
        "sources": [],
        "confidence": "low"
    }

def _safe_fail_message(reason: str, query: str, **kwargs) -> dict:
    """
    Generate safe failure message when retrieval fails or returns low-quality results.
    
    This ensures we NEVER fabricate answers when chunks are insufficient.
    """
    logger.warning(f"Retrieval failed: {reason}, query: '{query[:50]}...', details: {kwargs}")
    
    messages = {
        "no_chunks_retrieved": {
            "summary": "I could not find relevant information in the SRTR documentation.",
            "detail": ("I searched the available SRTR documentation but could not find content relevant to your question. "
                      "This might be because:\n"
                      "- The topic is not covered in the indexed documents\n"
                      "- The question needs to be phrased differently\n"
                      "- This requires accessing data tables rather than documentation\n\n"
                      "Please try:\n"
                      "- Rephrasing your question with more specific terms\n"
                      "- Asking about a different aspect of the topic\n"
                      "- Consulting SRTR's official website directly"),
        },
        "low_similarity": {
            "summary": "The retrieved information may not be relevant to your question.",
            "detail": (f"I found some content but with low confidence (similarity: {kwargs.get('max_sim', 0):.2f}). "
                      "This suggests the question might be outside the scope of indexed documentation or needs clarification. "
                      "Please try rephrasing your question or providing more context."),
        },
        "insufficient_quality_chunks": {
            "summary": "I found limited relevant information for your question.",
            "detail": (f"Only {kwargs.get('count', 0)} relevant sections were found, which may not be sufficient for a complete answer. "
                      "Please try:\n"
                      "- Being more specific about what aspect you're interested in\n"
                      "- Breaking down your question into smaller parts\n"
                      "- Asking about related but more general concepts first"),
        }
    }
    
    msg = messages.get(reason, messages["no_chunks_retrieved"])
    
    return {
        "summary": msg["summary"],
        "detail": msg["detail"],
        "sources": [],
        "confidence": "failed",
        "failure_reason": reason
    }

def _extract_calculator_parameters(query: str, tool_name: str) -> Dict[str, Any]:
    """
    Use LLM to extract structured parameters from natural language query.
    
    This is the key function that converts:
    "I'm 45 years old, type O, been on dialysis for 2 years, CPRA is 85"
    →
    {"age": 45, "blood_type": "O", "dialysis_time": 24, "cpra": 85}
    """
    
    if tool_name == "kidney_waiting_time":
        system = textwrap.dedent("""
        You are a parameter extraction assistant for the Kidney Waiting Time Calculator.
        
        Your task: Extract structured parameters from the user's natural language query.
        
        Required parameters:
        - blood_type: "O", "A", "B", or "AB"
        - age: Patient age in years (integer)
        - dialysis_time: Time on dialysis in MONTHS (integer)
        - cpra: CPRA percentage (0-100, integer)
        
        Optional parameters:
        - diabetes: true/false (default: false)
        - region: UNOS region 1-11 (default: 5)
        
        Important conversions:
        - Dialysis time: Convert years to months (e.g., "2 years" → 24 months)
        - Blood type: Normalize to uppercase single letter (e.g., "type o" → "O")
        - CPRA: May be expressed as percentage or decimal (e.g., "0.85" → 85)
        
        If a parameter is not mentioned, mark it as null (except optional parameters with defaults).
        
        Return ONLY JSON format, without any annotationS:
        {
          "blood_type": "O",
          "age": 45,
          "dialysis_time": 24,
          "cpra": 85,
          "diabetes": false,
          "region": 5,
          "extracted_info": {
            "dialysis_time_original": "2 years",
            "cpra_original": "85%"
          },
          "missing_parameters": []
        }
        
        Examples:
        
        Q: "I'm 45, type O, been on dialysis 2 years, CPRA is 85"
        A: {"blood_type": "O", "age": 45, "dialysis_time": 24, "cpra": 85, "diabetes": false, "region": 5, "extracted_info": {"dialysis_time_original": "2 years"}, "missing_parameters": []}
        
        Q: "Calculate wait time for a 30-year-old AB patient, 6 months dialysis, CPRA 5%, diabetic"
        A: {"blood_type": "AB", "age": 30, "dialysis_time": 6, "cpra": 5, "diabetes": true, "region": 5, "extracted_info": {}, "missing_parameters": []}
        
        Q: "Type B patient, 55 years old"
        A: {"blood_type": "B", "age": 55, "dialysis_time": null, "cpra": null, "diabetes": false, "region": 5, "extracted_info": {}, "missing_parameters": ["dialysis_time", "cpra"]}
        """).strip()
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query}
        ]
        
        try:
            raw_response = _openai_chat(messages, temperature=0.1, max_tokens=400)
            print(raw_response)
            params = json.loads(raw_response)
            
            logger.info(f"Extracted parameters: {params}")
            return params
            
        except Exception as e:
            logger.exception(f"Parameter extraction failed: {e}")
            return {
                "error": f"Could not extract parameters from query: {str(e)}",
                "query": query
            }
    
    else:
        return {"error": f"Unknown tool: {tool_name}"}

def _handle_calculator_query(query: str, plan: dict) -> dict:
    """
    Handle calculator/tool queries.
    
    Flow:
    1. Extract parameters from natural language
    2. Validate parameters
    3. Call appropriate calculator function
    4. Format results with explanation
    """
    
    tool_name = plan.get("entity_identifiers", {}).get("tool_name", "kidney_waiting_time")
    
    logger.info(f"Handling calculator query: tool={tool_name}")
    
    # Step 1: Extract parameters
    params = _extract_calculator_parameters(query, tool_name)
    
    if "error" in params:
        return {
            "summary": "I couldn't extract all required parameters from your query.",
            "detail": params["error"],
            "key_points": [],
            "sources": [],
            "confidence": "failed"
        }
    
    # Step 2: Check for missing parameters
    missing = params.get("missing_parameters", [])
    if missing:
        return {
            "summary": f"Please provide the following information: {', '.join(missing)}",
            "detail": generate_parameter_prompt(tool_name, missing, params),
            "key_points": [],
            "sources": [],
            "confidence": "incomplete",
            "missing_parameters": missing,
            "partial_parameters": {k: v for k, v in params.items() if v is not None and k not in ["missing_parameters", "extracted_info"]}
        }
    
    # Step 3: Call calculator
    if tool_name == "kidney_waiting_time":
        # Remove metadata fields before passing to calculator
        calc_params = {k: v for k, v in params.items() 
                      if k not in ["extracted_info", "missing_parameters", "error"]}
        
        result = calculate_kidney_waiting_time(calc_params)
        
        if "error" in result:
            return {
                "summary": "Calculation error.",
                "detail": result["error"],
                "key_points": [],
                "sources": [],
                "confidence": "error"
            }
        
        # Step 4: Format results
        return format_calculator_result(result, query)
    
    else:
        return {
            "summary": f"Calculator '{tool_name}' is not implemented yet.",
            "detail": "This tool is planned but not yet available.",
            "key_points": [],
            "sources": [],
            "confidence": "not_implemented"
        }

def generate_parameter_prompt(tool_name: str, missing: List[str], partial: dict) -> str:
    """Generate helpful prompt for missing parameters."""
    
    prompts = {
        "blood_type": "Your blood type (O, A, B, or AB)",
        "age": "Your current age in years",
        "dialysis_time": "How long you've been on dialysis (in months or years)",
        "cpra": "Your CPRA (Calculated Panel Reactive Antibody) percentage, typically 0-100. If you don't know this, you can estimate it as 0 for non-sensitized patients."
    }
    
    details = f"I found: {', '.join(f'{k}={v}' for k, v in partial.items() if k not in ['missing_parameters', 'extracted_info'])}\n\n"
    details += "To calculate your estimated waiting time, I still need:\n\n"
    
    for param in missing:
        details += f"• **{param}**: {prompts.get(param, 'This parameter')}\n"
    
    details += "\nPlease provide this information and I'll calculate your estimated waiting time."
    
    return details

def format_calculator_result(result: dict, original_query: str) -> dict:
    """Format calculator result into structured response."""
    
    # Summary
    summary = f"Estimated median waiting time: **{result['median_wait_readable']}**"
    
    # Detailed explanation
    detail_parts = []
    
    # Profile
    profile = result["patient_profile"]
    detail_parts.append(
        f"**Patient Profile:**\n"
        f"- Blood Type: {profile['blood_type']}\n"
        f"- Age: {profile['age']} years\n"
        f"- Dialysis Time: {profile['dialysis_time_months']} months\n"
        f"- CPRA: {profile['cpra']}%\n"
        f"- Diabetes: {'Yes' if profile.get('diabetes') else 'No'}\n"
        f"- Region: {profile.get('region', 'N/A')}"
    )
    
    # Wait time range
    ranges = result["wait_ranges"]
    detail_parts.append(
        f"\n**Waiting Time Estimates:**\n"
        f"- 25% of similar patients wait: **{ranges['25th_percentile_readable']}** or less\n"
        f"- 50% (median) wait: **{ranges['median_readable']}**\n"
        f"- 75% wait: **{ranges['75th_percentile_readable']}** or less"
    )
    
    # Key factors
    if result.get("factors_impact"):
        detail_parts.append("\n**Factors Affecting Your Wait Time:**")
        for factor_name, factor_data in result["factors_impact"].items():
            impact = factor_data.get("impact", "")
            reason = factor_data.get("reason", "")
            detail_parts.append(f"- **{factor_name.replace('_', ' ').title()}**: {impact} — {reason}")
    
    # Interpretation
    if result.get("interpretation"):
        detail_parts.append(f"\n**Interpretation:**\n{result['interpretation']}")
    
    # Notes and disclaimer
    detail_parts.append(f"\n**Notes:**\n{result.get('calculation_notes', '')}")
    detail_parts.append(f"\n⚠️ {result.get('disclaimer', '')}")
    
    detail = "\n".join(detail_parts)
    
    # Key points
    key_points = [
        f"Median wait: {result['median_wait_readable']}",
        f"Range: {ranges['25th_percentile_readable']} to {ranges['75th_percentile_readable']}",
    ]
    
    # Add most impactful factors
    if result.get("factors_impact"):
        for factor_name, factor_data in result["factors_impact"].items():
            if "+" in factor_data.get("impact", ""):
                key_points.append(f"{factor_name.title()}: {factor_data['impact']}")
    
    return {
        "summary": summary,
        "detail": detail,
        "key_points": key_points[:5],  # Limit to 5 key points
        "calculation_result": result,
        "sources": [{
            "section": "Kidney Waiting Time Calculator",
            "doc_type": "calculator_tool",
            "url": "https://www.srtr.org/tools/kidney-transplant-waiting-times/",
            "note": "Simplified demonstration model"
        }],
        "confidence": "high",
        "tool_used": "kidney_waiting_time_calculator"
    }

def _handle_numeric_query(query: str, plan: dict) -> dict:
    """
    Handle numeric queries using deterministic tools.
    
    TODO: Implement actual database/CSV lookups for center statistics.
    For now, returns a placeholder explaining the approach.
    """
    entities = plan.get("entity_identifiers", {})
    
    return {
        "summary": "Numeric query handler not yet implemented.",
        "detail": (f"This query requires looking up specific numeric data:\n"
                  f"- Intent: {plan.get('intent')}\n"
                  f"- Entities: {json.dumps(entities, indent=2)}\n\n"
                  f"This should be handled by:\n"
                  f"1. Querying local CSV/database for exact values\n"
                  f"2. Applying appropriate filters (center, organ, time period)\n"
                  f"3. Returning precise numbers with confidence intervals\n"
                  f"4. Adding context from RAG for interpretation\n\n"
                  f"Status: Implementation pending"),
        "sources": [],
        "confidence": "not_implemented",
        "entities": entities
    }

def _generate_answer_from_chunks(query: str, chunks: List[Dict[str, Any]]) -> dict:
    """
    Generate answer strictly based on retrieved chunks.
    
    Enhanced output with summary + detail structure for better UX.
    """
    system = textwrap.dedent("""
    You are Agent B - SRTR Documentation Assistant.
    
    CRITICAL RULES:
    1. You may ONLY answer based on the provided text chunks
    2. If chunks don't contain necessary information, you MUST say "Based on the provided SRTR documentation, I cannot determine..."
    3. NEVER fabricate methods, formulas, or metric definitions
    4. Rephrase information in your own words, but keep core definitions and numbers accurate
    5. Output in a structured format with both summary and detail
    
    Output format (JSON):
    {
      "summary": "One clear sentence answering the core question",
      "detail": "Comprehensive explanation with context and nuances",
      "key_points": ["point 1", "point 2", ...],  // optional: main takeaways
      "sources_used": ["Section 1", "Section 2"]
    }
    
    Guidelines:
    - summary: 1-2 sentences maximum, directly answers the question
    - detail: 2-4 paragraphs providing full context, methodology, caveats
    - key_points: Bullet points for complex topics (optional)
    - sources_used: List section titles you actually used
    
    Example:
    Q: "What is 1-year patient survival?"
    A: {
      "summary": "1-year patient survival measures the percentage of transplant recipients who are alive one year after their transplant.",
      "detail": "This metric is one of SRTR's core outcome measures used to evaluate transplant center performance. It includes all patients regardless of graft function, meaning even patients who lost their graft but are still alive are counted. The metric is risk-adjusted to account for differences in patient populations across centers. SRTR calculates expected survival rates based on patient characteristics and compares actual outcomes to these predictions.",
      "key_points": [
        "Includes all patients alive at 1 year, regardless of graft status",
        "Risk-adjusted for fair comparison across centers",
        "Core metric for center performance evaluation"
      ],
      "sources_used": ["1-year Patient Survival Definition", "Risk Adjustment Methodology"]
    }
    """).strip()
    
    # Format chunks for LLM
    chunks_text = "\n\n---\n\n".join([
        f"Chunk {i+1} (from '{chunk['section_title'] or 'unknown section'}'):\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    ])
    
    user_content = f"""Question: {query}

Available information from SRTR documentation:

{chunks_text}

Please answer the question based solely on the above chunks. Return your answer in the specified JSON format."""
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]
    
    try:
        answer_json = _openai_chat(messages, temperature=0.2, max_tokens=1800)
        
        # Parse JSON response
        try:
            parsed = json.loads(answer_json)
            summary = parsed.get("summary", "")
            detail = parsed.get("detail", "")
            key_points = parsed.get("key_points", [])
            sources_used = parsed.get("sources_used", [])
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            summary = answer_json[:200] + "..." if len(answer_json) > 200 else answer_json
            detail = answer_json
            key_points = []
            sources_used = []
        
        # Build full sources list with similarity scores
        sources = [
            {
                "section": chunk["section_title"],
                "doc_type": chunk["doc_type"],
                "url": chunk.get("source_url", ""),
                "similarity": chunk["similarity"],
                "used_in_answer": chunk["section_title"] in sources_used if sources_used else True
            }
            for chunk in chunks
        ]
        
        return {
            "summary": summary,
            "detail": detail,
            "key_points": key_points,
            "sources": sources,
            "confidence": "high" if chunks[0]["similarity"] > 0.75 else "medium"
        }
        
    except Exception as e:
        logger.exception(f"Answer generation failed: {e}")
        return {
            "summary": "Error generating answer.",
            "detail": f"An error occurred while processing the retrieved information: {str(e)}",
            "key_points": [],
            "sources": [],
            "confidence": "error"
        }

# ==========
# HTTP Endpoints
# ==========


@csrf_exempt
def api_query(request):
    """
    POST /api/query
    Main query endpoint with enhanced RAG pipeline.
    
    Returns structured response with summary + detail.
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
        # Step 1: Initial Agent - enhanced intent classification
        plan = _initial_agent_plan(user_prompt)
        logger.info(f"Initial Agent: intent={plan.get('intent')}, "
                   f"mode={plan.get('answer_mode')}, "
                   f"entities={plan.get('entity_identifiers')}")

        # Step 2: Agent B - generate answer with enhanced structure
        result = _agent_b_generate_answer(user_prompt, plan)
        print(result)
        logger.info(f"Agent B: confidence={result.get('confidence')}")

        # Format response with enhanced structure
        resp = {
            # Core answer
            "summary": result.get("summary", ""),
            "detail": result.get("detail", ""),
            "key_points": result.get("key_points", []),
            
            # Legacy field for backward compatibility
            "answer_text": result.get("summary", ""),
            
            # Source attribution
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", "unknown"),
            
            # Plan details for debugging/transparency
            "plan": {
                "intent": plan.get("intent"),
                "answer_mode": plan.get("answer_mode"),
                "entity_identifiers": plan.get("entity_identifiers"),
                "semantic_scope": plan.get("semantic_scope"),
                "filters": plan.get("filters"),
                "rationale": plan.get("rationale")
            },
            
            # Metadata
            "metadata": {
                "retrieval_needed": plan.get("retrieval_needed"),
                "use_deterministic_tool": plan.get("use_deterministic_tool"),
                "failure_reason": result.get("failure_reason")
            }
        }
        
        return JsonResponse(resp)
    
    except Exception as e:
        logger.exception("Error in api_query")
        return JsonResponse({
            "error": str(e),
            "summary": "An error occurred while processing your request.",
            "detail": f"Error details: {str(e)}",
            "confidence": "error"
        }, status=500)

@csrf_exempt
def api_debug(request):
    """
    GET /api/debug?q=...
    Debug endpoint to test retrieval without full pipeline.
    """
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    query = request.GET.get("q", "")
    if not query:
        return JsonResponse({"error": "No query provided"}, status=400)
    
    try:
        # Test retrieval
        chunks = retrieve_chunks(query, filters=None, top_k=5)
        
        return JsonResponse({
            "query": query,
            "retrieved_chunks": len(chunks),
            "chunks": [
                {
                    "text": chunk["text"][:200] + "...",
                    "section": chunk["section_title"],
                    "similarity": chunk["similarity"]
                }
                for chunk in chunks
            ]
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
@csrf_exempt
def api_calculate(request):
    """
    POST /api/calculate
    Direct calculator endpoint for testing.
    """
    # 1. 方法检查
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    # 2. JSON解析
    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tool = data.get("tool", "kidney_waiting_time")
    params = {}
    
    # 3. 参数模式判断
    if "parameters" in data:
        # Mode 1: Direct parameters
        params = data["parameters"]
    elif "query" in data:
        # Mode 2: Extract from natural language
        params = _extract_calculator_parameters(data["query"], tool)
        if "error" in params:
            return JsonResponse(params, status=400)
    else:
        # 错误：缺少参数或查询
        return JsonResponse({"error": "Must provide either 'parameters' or 'query'"}, status=400)
    
    # 4. 调用计算器工具
    if tool == "kidney_waiting_time":
        # 过滤掉不属于计算器参数的元数据
        calc_params = {k: v for k, v in params.items() 
                       if k not in ["extracted_info", "missing_parameters", "error"]}
        
        # 假设 calculate_kidney_waiting_time, JsonResponse, 等已导入
        result = calculate_kidney_waiting_time(calc_params)
        
        if "error" in result:
            return JsonResponse(result, status=400)
        
        return JsonResponse({
            "tool": tool,
            "input_parameters": calc_params,
            "result": result,
            "status": "success"
        })
    else:
        # 错误：工具未知
        return JsonResponse({"error": f"Unknown tool: {tool}"}, status=400)

# 假设 load_documents_index, load_chunks_index, logger 已导入
@csrf_exempt
def api_rebuild_index(request):
    """
    POST /api/rebuild_index
    Rebuild document and chunk indexes (admin function).
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        # TODO: Implement full rebuild logic (e.g., calling the function that performs the rebuild)
        
        # 这里的代码似乎只是在获取索引统计信息，而不是重建
        docs_index = load_documents_index()
        chunks_index = load_chunks_index()
        
        return JsonResponse({
            "status": "ok",
            "message": "Index information retrieved",
            "stats": {
                "documents": len(docs_index),
                "chunks": len(chunks_index)
            }
        })
    except Exception as e:
        logger.exception("Error in rebuild_index")
        return JsonResponse({
            "error": str(e),
            "status": "error"
        }, status=500)
    

# ==========
# 管理端点
# ==========

"""
@csrf_exempt
def api_execute(request):
    #POST /api/execute
    #Input: { "runner": "r"|"python", "code": "...", "workdir": "..." }
    #Return: { "status": "ok", "artifacts": [...], "logs": "..." }
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
        }, status=500)"""

"""@csrf_exempt
def api_rebuild_index(request):
    #POST /api/rebuild_index
    #手动重建字典索引
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
        }, status=500)"""

"""@csrf_exempt
def api_debug(request):
    #GET /api/debug?q=CAN_ABO
    #调试端点：显示变量提取和路径匹配的详细信息
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
        }, status=500)"""
    
