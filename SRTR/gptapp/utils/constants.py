# ==========
# Path Configuration
# ==========
from pathlib import Path

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