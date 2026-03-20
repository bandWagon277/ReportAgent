# ==========
# RAG Retrieval Pipeline
# ==========

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json
from utils.constants import EMBEDDINGS_DIR
from services.llm import LLMService
import csv
from io import StringIO

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    RAGEngine: The core engine responsible for vector retrieval.
    It relies on LLMService to compute the vector representation of the user's query,
    and then calculates the cosine similarity with local document chunks.
    """

    def __init__(self, llm_service: LLMService):

        self.llm = llm_service

    def retrieve_chunks(self, query: str, chunks_data: Dict,filters: Optional[Dict[str, Any]] = None, 
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
            chunks_index = chunks_data
            
            if not chunks_index:
                logger.warning("No chunks available for retrieval")
                return []
            
            # Stage 1: Pre-filter by semantic scope and metadata
            candidate_chunks = []
            
            for chunk_id, chunk in chunks_index.items():
                # Apply metadata filters
                if filters and not self._matches_filters(chunk, filters):
                    continue
                
                # Apply semantic scope filtering (keyword-based)
                if semantic_scope and not self._matches_semantic_scope(chunk, semantic_scope):
                    continue
                
                candidate_chunks.append((chunk_id, chunk))
            
            if not candidate_chunks:
                logger.warning(f"No chunks passed pre-filtering. Filters: {filters}, Scope: {semantic_scope}")
                return []
            
            logger.info(f"Pre-filtering: {len(candidate_chunks)} candidates from {len(chunks_index)} total chunks")
            
            # Stage 2: Semantic similarity search
            llm_service = self.llm
            query_embedding = llm_service.get_embedding(query)
            scored_chunks = []
            
            for chunk_id, chunk in candidate_chunks:
                # Load chunk embedding
                embedding_path = EMBEDDINGS_DIR / f"{chunk_id}.json"
                if not embedding_path.exists():
                    continue
                
                with open(embedding_path, 'r') as f:
                    chunk_embedding = json.load(f)["embedding"]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
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

    def _matches_semantic_scope(self,chunk: Dict[str, Any], semantic_scope: List[str]) -> bool:
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

    def _matches_filters(self,chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
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

    def _cosine_similarity(self,vec1: List[float], vec2: List[float]) -> float:
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
