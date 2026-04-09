"""
agents.py — New agent modules inspired by three HKUST papers:
  - LAMBDA: Self-correction loop (Inspector Agent)
  - DARE: Distribution-aware data profiling
  - Survey: Knowledge Integration Mechanism (KIM)

Architecture follows SRTR's service-layer pattern:
  DataProfiler  → rich dataset metadata for every LLM prompt
  InspectorAgent → retry failed code with error context (up to N attempts)
  KnowledgeBase  → embed + retrieve positive feedback as few-shot examples
"""

import json
import logging
import math
import os
import re
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)


# ============================================================
# 1. DATA PROFILER  (DARE-inspired)
# ============================================================
# Generates a rich data profile including:
#   - General info: rows, columns, column types
#   - Statistical characteristics: distribution assumptions,
#     dimensionality, feature types, sparsity
#   - Per-column details: missing %, unique count, top values
# This profile is injected into every LLM prompt so the model
# can make distribution-aware method selections.

class DataProfiler:
    """Build a structured data profile from an uploaded CSV."""

    # Thresholds
    HIGH_DIM_THRESHOLD = 50          # columns above this = "high" dimensionality
    SPARSE_THRESHOLD = 0.3           # >30% missing = sparse
    CATEGORICAL_NUNIQUE_RATIO = 0.05 # unique/rows < 5% → likely categorical

    @classmethod
    def profile(cls, csv_path: str) -> Dict[str, Any]:
        """
        Main entry point. Returns a dict with:
          general_info, statistical_characteristics, column_profiles
        """
        try:
            with default_storage.open(csv_path, 'r') as f:
                csv_text = f.read()
            df = pd.read_csv(StringIO(csv_text))
        except Exception as e:
            logger.error(f"DataProfiler: failed to read CSV: {e}")
            return {"error": str(e)}

        profile = {
            "general_info": cls._general_info(df),
            "statistical_characteristics": cls._statistical_characteristics(df),
            "column_profiles": cls._column_profiles(df),
        }
        return profile

    @classmethod
    def profile_to_prompt(cls, csv_path: str) -> str:
        """Return a text block ready to inject into an LLM prompt."""
        p = cls.profile(csv_path)
        if "error" in p:
            return f"[Data profiling error: {p['error']}]"

        gi = p["general_info"]
        sc = p["statistical_characteristics"]
        cols = p["column_profiles"]

        lines = [
            "### DATA PROFILE (auto-generated)",
            f"Rows: {gi['num_rows']}  |  Columns: {gi['num_features']}  |  "
            f"Missing total: {gi['missing_total']}",
            f"Data modality: {sc['data_modality']}  |  "
            f"Dimensionality: {sc['dimensionality']}  |  "
            f"Primary feature type: {sc['primary_feature_type']}",
            f"Distribution assumption: {sc['distribution_assumption']}  |  "
            f"Sparsity: {sc['sparsity_structure']}",
            "",
            "Column details:",
        ]
        for cp in cols[:60]:  # cap at 60 columns
            miss_pct = f"{cp['missing_pct']:.1f}%"
            lines.append(
                f"  - {cp['name']} ({cp['dtype']}): "
                f"unique={cp['n_unique']}, missing={miss_pct}"
                + (f", top={cp['top_values']}" if cp.get('top_values') else "")
                + (f", mean={cp['mean']:.2f}, std={cp['std']:.2f}"
                   if cp.get('mean') is not None else "")
            )
        return "\n".join(lines)

    # ---------- internals ----------

    @classmethod
    def _general_info(cls, df: pd.DataFrame) -> Dict:
        return {
            "num_rows": int(len(df)),
            "num_features": int(df.shape[1]),
            "features": df.columns.tolist(),
            "missing_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    @classmethod
    def _statistical_characteristics(cls, df: pd.DataFrame) -> Dict:
        n_rows, n_cols = df.shape
        num_cols = df.select_dtypes(include='number')
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool'])

        # Dimensionality
        dimensionality = "high" if n_cols > cls.HIGH_DIM_THRESHOLD else "low"

        # Primary feature type
        if num_cols.shape[1] > cat_cols.shape[1]:
            primary_ft = "numerical"
        elif cat_cols.shape[1] > num_cols.shape[1]:
            primary_ft = "categorical"
        else:
            primary_ft = "mixed"

        # Distribution assumption (test normality on numeric cols sample)
        dist_assumption = "unknown"
        if num_cols.shape[1] > 0:
            try:
                from scipy import stats as sp_stats
                normal_count = 0
                test_cols = num_cols.columns[:20]  # test up to 20 cols
                for col in test_cols:
                    vals = num_cols[col].dropna()
                    if len(vals) >= 8:
                        _, p_value = sp_stats.shapiro(
                            vals.sample(min(500, len(vals)), random_state=42)
                        )
                        if p_value > 0.05:
                            normal_count += 1
                ratio = normal_count / max(len(test_cols), 1)
                dist_assumption = "gaussian" if ratio > 0.5 else "non-gaussian"
            except Exception:
                dist_assumption = "non-gaussian"

        # Sparsity
        missing_ratio = df.isna().sum().sum() / max(n_rows * n_cols, 1)
        sparsity = "sparse" if missing_ratio > cls.SPARSE_THRESHOLD else "dense"

        return {
            "data_modality": "tabular",
            "dimensionality": dimensionality,
            "primary_feature_type": primary_ft,
            "distribution_assumption": dist_assumption,
            "sparsity_structure": sparsity,
            "num_numeric_cols": int(num_cols.shape[1]),
            "num_categorical_cols": int(cat_cols.shape[1]),
        }

    @classmethod
    def _column_profiles(cls, df: pd.DataFrame) -> List[Dict]:
        profiles = []
        for col in df.columns:
            series = df[col]
            cp: Dict[str, Any] = {
                "name": col,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique()),
                "missing_pct": float(series.isna().mean() * 100),
            }
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe()
                cp["mean"] = float(desc.get("mean", 0))
                cp["std"] = float(desc.get("std", 0))
                cp["min"] = float(desc.get("min", 0))
                cp["max"] = float(desc.get("max", 0))
            else:
                top = series.value_counts(dropna=True).head(5)
                cp["top_values"] = {str(k): int(v) for k, v in top.items()}
            profiles.append(cp)
        return profiles


# ============================================================
# 2. INSPECTOR AGENT  (LAMBDA-inspired self-correction loop)
# ============================================================
# When generated code fails execution, the Inspector:
#   1. Receives the failed code + error traceback
#   2. Asks the LLM to diagnose and fix
#   3. Returns corrected code
# The caller retries up to max_retries times.
# This is LAMBDA's single biggest reliability improvement.

class InspectorAgent:
    """Self-correction loop: diagnose code execution errors and produce fixes."""

    SYSTEM_PROMPT = (
        "You are a code inspector and debugger. "
        "You receive Python code that failed during execution, along with the "
        "error traceback and the dataset context. "
        "Your job is to:\n"
        "1. Diagnose the root cause of the error.\n"
        "2. Produce a CORRECTED version of the entire Python code.\n"
        "3. Return ONLY one fenced ```python block with the fixed code.\n"
        "Do NOT explain the fix in prose — just return the corrected code block.\n"
        "Common issues to look for:\n"
        "- Column name mismatches (check the dataset context for exact names)\n"
        "- Type errors (e.g., string vs numeric operations)\n"
        "- Missing imports\n"
        "- Division by zero or empty DataFrame operations\n"
        "- Incorrect variable names (e.g., 'processed_data' not set)\n"
    )

    @classmethod
    def inspect_and_fix(
        cls,
        failed_code: str,
        error_message: str,
        data_context: str,
        api_key: str,
        output_type: str = "csv",
    ) -> str:
        """
        Send the failed code + error to the LLM and get a corrected version.
        Returns the corrected Python code string (extracted from fenced block).
        Raises RuntimeError if the LLM cannot produce a fix.
        """
        from .gpt_backend_utils import call_openai_chat, extract_python_code

        user_msg = (
            f"## Failed Code\n```python\n{failed_code}\n```\n\n"
            f"## Error Traceback\n```\n{error_message[:2000]}\n```\n\n"
            f"## Dataset Context\n{data_context[:3000]}\n\n"
            f"## Output Type: {output_type}\n"
            "Please fix the code and return the complete corrected version "
            "in a single ```python block."
        )

        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        raw_response = call_openai_chat(api_key, messages, temperature=0.1, timeout=90)
        fixed_code = extract_python_code(raw_response)

        if not fixed_code:
            raise RuntimeError("Inspector could not produce corrected code.")

        return fixed_code

    @classmethod
    def execute_with_retry(
        cls,
        code: str,
        csv_path: str,
        output_type: str,
        api_key: str,
        data_context: str = "",
        image_path: str = None,
        max_retries: int = 3,
        dry_run: bool = False,
        row_limit: int = None,
    ) -> Tuple[Any, str, int]:
        """
        Execute code with automatic self-correction on failure.

        Returns: (result, final_code, attempts_used)
          - result: the execution output (CSV text, image bytes, PDF bytes, artifacts dict, or Exception)
          - final_code: the code that eventually succeeded (or last attempted)
          - attempts_used: how many attempts were made (1 = first try succeeded)
        """
        from .gpt_backend_utils import (
            execute_python_code, save_to_file, sanitize_python,
        )

        current_code = code
        last_error = None

        for attempt in range(1, max_retries + 1):
            logger.info(f"InspectorAgent: attempt {attempt}/{max_retries} "
                        f"for {output_type}")

            # Save current code to file
            code_path = save_to_file(f"inspector_attempt_{attempt}.py",
                                     sanitize_python(current_code))

            result = execute_python_code(
                csv_file_path=csv_path,
                py_file_path=code_path,
                output_type=output_type,
                csv_path=csv_path,
                image_path=image_path,
                dry_run=dry_run,
                row_limit=row_limit,
            )

            # Success
            if not isinstance(result, Exception):
                if dry_run and isinstance(result, dict) and not result.get('ok'):
                    last_error = result.get('error', 'Unknown dry-run error')
                else:
                    logger.info(f"InspectorAgent: succeeded on attempt {attempt}")
                    return result, current_code, attempt

            # Failure — extract error message
            if isinstance(result, Exception):
                last_error = str(result)
            elif isinstance(result, dict):
                last_error = result.get('error', str(result))
            else:
                last_error = str(result)

            logger.warning(f"InspectorAgent: attempt {attempt} failed: "
                           f"{last_error[:200]}")

            # Don't retry on last attempt
            if attempt >= max_retries:
                break

            # Ask Inspector to fix
            try:
                current_code = cls.inspect_and_fix(
                    failed_code=current_code,
                    error_message=last_error,
                    data_context=data_context,
                    api_key=api_key,
                    output_type=output_type,
                )
                logger.info(f"InspectorAgent: received fix, will retry")
            except Exception as fix_err:
                logger.error(f"InspectorAgent: fix generation failed: {fix_err}")
                break

        # All retries exhausted
        logger.error(f"InspectorAgent: all {max_retries} attempts failed")
        return (Exception(last_error) if last_error
                else Exception("Max retries exhausted")), current_code, max_retries


# ============================================================
# 3. KNOWLEDGE INTEGRATION MECHANISM  (LAMBDA KIM + SRTR RAG)
# ============================================================
# A lightweight knowledge base that:
#   - Indexes positive-rated feedback (query → code) with embeddings
#   - Retrieves the most similar past examples for few-shot context
#   - Stores embeddings as JSON files (like SRTR's data_repo pattern)
#   - Falls back gracefully if no embeddings / no OpenAI key

class KnowledgeBase:
    """
    Embedding-based retrieval of successful analysis patterns
    from user feedback history.

    Storage layout (under MEDIA_ROOT/knowledge_base/):
      index.json         — master index: [{id, query, code_summary, output_type,
                                            embedding_file}, ...]
      embeddings/
        fb_<id>.json     — {"embedding": [float, ...]}
    """

    # Class-level defaults
    SIMILARITY_THRESHOLD = 0.70
    INDEX_CACHE_KEY = "kim:index"
    INDEX_CACHE_TTL = 600  # 10 min

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(
                getattr(settings, 'MEDIA_ROOT', ''), 'knowledge_base'
            )
        self.base_dir = base_dir
        self.embeddings_dir = os.path.join(base_dir, "embeddings")
        self.index_path = os.path.join(base_dir, "index.json")
        os.makedirs(self.embeddings_dir, exist_ok=True)

    # ---------- Index management ----------

    def load_index(self) -> List[Dict]:
        """Load the knowledge base index (with cache)."""
        cached = cache.get(self.INDEX_CACHE_KEY)
        if cached is not None:
            return cached

        if not os.path.exists(self.index_path):
            return []

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            cache.set(self.INDEX_CACHE_KEY, index, self.INDEX_CACHE_TTL)
            return index
        except Exception as e:
            logger.error(f"KnowledgeBase: failed to load index: {e}")
            return []

    def save_index(self, index: List[Dict]):
        """Persist the index to disk and invalidate cache."""
        try:
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            cache.set(self.INDEX_CACHE_KEY, index, self.INDEX_CACHE_TTL)
        except Exception as e:
            logger.error(f"KnowledgeBase: failed to save index: {e}")

    # ---------- Embedding helpers ----------

    @staticmethod
    def _get_embedding(text: str, api_key: str) -> Optional[List[float]]:
        """Call OpenAI embeddings API. Returns None on failure."""
        import requests as req
        try:
            resp = req.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("EMBEDDING_MODEL",
                                       "text-embedding-3-small"),
                    "input": text[:8000],
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(f"Embedding API error: {resp.status_code}")
                return None
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Embedding call failed: {e}")
            return None

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def _load_embedding(self, entry_id: int) -> Optional[List[float]]:
        path = os.path.join(self.embeddings_dir, f"fb_{entry_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)["embedding"]
        except Exception:
            return None

    def _save_embedding(self, entry_id: int, embedding: List[float]):
        path = os.path.join(self.embeddings_dir, f"fb_{entry_id}.json")
        try:
            with open(path, 'w') as f:
                json.dump({"embedding": embedding}, f)
        except Exception as e:
            logger.error(f"KnowledgeBase: failed to save embedding: {e}")

    # ---------- Build / update ----------

    def rebuild_from_feedback(self, api_key: str) -> int:
        """
        Scan Feedback table for positive entries and build/update the
        knowledge base index + embeddings. Returns count of indexed entries.
        """
        from .models import Feedback

        positive = Feedback.objects.filter(rating='positive').order_by('-created_at')[:200]
        if not positive.exists():
            logger.info("KnowledgeBase: no positive feedback to index")
            return 0

        index = self.load_index()
        existing_ids = {entry["id"] for entry in index}
        added = 0

        for fb in positive:
            if fb.pk in existing_ids:
                continue

            # Build the text to embed: query + summary
            text_to_embed = (
                f"Query: {fb.query_text}\n"
                f"Output type: {fb.output_type}\n"
                f"Approach: {fb.response_summary[:500]}"
            )

            embedding = self._get_embedding(text_to_embed, api_key)
            if embedding is None:
                continue

            self._save_embedding(fb.pk, embedding)

            index.append({
                "id": fb.pk,
                "query_text": fb.query_text[:500],
                "code_summary": fb.response_summary[:500],
                "output_type": fb.output_type,
                "comment": (fb.comment or "")[:200],
                "embedding_file": f"fb_{fb.pk}.json",
            })
            existing_ids.add(fb.pk)
            added += 1

        self.save_index(index)
        logger.info(f"KnowledgeBase: indexed {added} new entries "
                    f"(total: {len(index)})")
        return added

    # ---------- Retrieval ----------

    def retrieve(
        self,
        query: str,
        api_key: str,
        output_type: str = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Find the top-k most similar past successful analyses for the query.
        Returns list of dicts with query_text, code_summary, similarity.
        """
        index = self.load_index()
        if not index:
            return []

        query_embedding = self._get_embedding(query, api_key)
        if query_embedding is None:
            return []

        scored = []
        for entry in index:
            # Optional: filter by output type
            if output_type and entry.get("output_type") != output_type:
                continue

            stored_emb = self._load_embedding(entry["id"])
            if stored_emb is None:
                continue

            sim = self._cosine_similarity(query_embedding, stored_emb)
            if sim >= self.SIMILARITY_THRESHOLD:
                scored.append({**entry, "similarity": round(sim, 4)})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def retrieve_as_prompt(
        self,
        query: str,
        api_key: str,
        output_type: str = None,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve similar examples and format them as a prompt section
        for few-shot context injection.
        """
        results = self.retrieve(query, api_key, output_type, top_k)
        if not results:
            return ""

        lines = [
            "\n### Similar Successful Analyses (Knowledge Base)",
            "The following past analyses were rated positively by users "
            "and are semantically similar to your current request. "
            "Use them as reference for approach and style:\n",
        ]
        for i, r in enumerate(results, 1):
            lines.append(
                f"**Example {i}** (similarity: {r['similarity']:.2f}):\n"
                f"  Query: \"{r['query_text'][:200]}\"\n"
                f"  Approach: {r['code_summary'][:300]}"
            )
            if r.get("comment"):
                lines.append(f"  User note: {r['comment'][:150]}")
            lines.append("")

        return "\n".join(lines)
