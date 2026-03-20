# ==========
# Dictionary Index Building & Loading 
# ==========
from utils.constants import DICT_INDEX, DATA_REPO,DICT_ROOT, DOCS_INDEX_PATH,CHUNKS_INDEX_PATH
import logging
from io import StringIO
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管家：负责所有本地文件的读取、索引查找和解析。
    MainAgent 不需要知道文件存在哪里，只需要问 DataManager 要数据。
    """

    def __init__(self):
        # 初始化时，我们可以预加载索引，或者什么都不做，等用到再加载
        # 这里为了简单，我们先不做耗时操作
        pass

    def build_dictionaries_index(self):
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

    def load_dictionaries_index(self):
        """Load dictionary index, build if not exists"""
        if not DICT_INDEX.exists():
            return self.build_dictionaries_index()
        
        try:
            with open(DICT_INDEX, "r", encoding="utf-8") as f:
                index = json.load(f)
                # Check index version, rebuild if outdated
                if "total_variables" not in index:
                    logger.info("Old index format detected, rebuilding...")
                    return self.build_dictionaries_index()
                return index
        except Exception as e:
            logger.error(f"Failed to load index, rebuilding: {e}")
            return self.build_dictionaries_index()
        
    def guess_paths_for_variable(self,var_name: str, index: dict, topk: int = 3) -> List[str]:
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
    # Document and Chunk Index Management
    # ==========

    def load_documents_index(self) -> Dict[str, Dict[str, Any]]:
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

    def load_chunks_index(self) -> Dict[str, Dict[str, Any]]:
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

    def save_chunks_index(self,chunks_index: Dict[str, Dict[str, Any]]):
        """Save chunks index to file."""
        CHUNKS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunks_index, f, ensure_ascii=False, indent=2)

    def _extract_concept_contents(self, concept_keywords: List[str], paths: List[str]) -> Dict[str, str]:
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
