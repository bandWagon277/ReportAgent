Initialization:
C:\Users\18120\Desktop\OPENAIproj\myenv\Scripts\activate.bat #activate environment
#python -m pip install requests
cd Desktop\OPENAIproj\SRTR
python manage.py runserver
(quit with: control C)
open link:  http://127.0.0.1:8000/gpt-interface/ 

Workflow
Overview
The system has been upgraded from local file reading to a RAG (Retrieval Augmented Generation) pipeline that works with HTML mirrors of SRTR web pages.
Architecture Flow
User Query
    ↓
[Initial Agent] - Intent Classification
    ↓
[RAG Retrieval] - Semantic Search (if needed)
    ↓
[Agent B] - Answer Generation (strict: only from retrieved chunks)
    ↓
Response with Sources

Directory Structure
data_repo/
├── html_mirrors/          # HTML mirror files from SRTR
│   ├── metrics_definitions.html
│   ├── wait_time_calculator.html
│   ├── center_UCLA_kidney.html
│   └── ...
│
├── chunks/                # Parsed text chunks (optional cache)
│
├── embeddings/            # Vector embeddings for each chunk
│   ├── ch_001.json
│   ├── ch_002.json
│   └── ...
│
├── meta/
│   ├── documents.index.json    # Document metadata
│   └── chunks.index.json       # Chunk metadata with text
│
├── dictionaries/          # Legacy: data dictionary CSVs
└── concepts/              # Legacy: concept calculation scripts

Data Flow
1. Offline: Build RAG Index
bashpython build_rag_index.py
What it does:

Scans html_mirrors/ directory
Detects document type from filename
Parses HTML → structured data
Splits into chunks (semantic sections)
Generates embeddings for each chunk
Saves to indexes

Output:

documents.index.json: Document metadata
chunks.index.json: All chunks with text and metadata
embeddings/*.json: Vector embeddings

2. Online: Query Processing
Step 1: Initial Agent (Intent Classification)
Input: User query
Output:
json{
  "intent": "metric_definition",
  "filters": {"doc_type": "metric_definition"},
  "retrieval_needed": true,
  "rationale": "User wants metric definition"
}
Intent Types:

metric_definition: What does metric X mean?
center_comparison: Compare centers A and B
wait_time: How is wait time calculated?
methodology: How does SRTR collect data?
data_dictionary: Variable definition lookup
general: General transplant question

Step 2: RAG Retrieval (Conditional)
When triggered: retrieval_needed = true
Process:

Generate query embedding
Calculate cosine similarity with all chunk embeddings
Apply filters (doc_type, organ, center_id, etc.)
Return top-K most similar chunks

Output:
json[
  {
    "chunk_id": "ch_001",
    "text": "1-year patient survival is defined as...",
    "section_title": "1-year Patient Survival",
    "doc_type": "metric_definition",
    "source_url": "https://www.srtr.org/...",
    "similarity": 0.87
  },
  ...
]
Step 3: Agent B (Answer Generation)
Critical Rules:

✅ ONLY use information from retrieved chunks
✅ Rephrase in own words, but keep facts accurate
❌ NEVER fabricate formulas, metrics, or methods
✅ If insufficient info: "Based on available SRTR documentation, I cannot determine..."
✅ Always cite sources at end

Output:
json{
  "answer_text": "...",
  "sources": [
    {
      "section": "1-year Patient Survival",
      "doc_type": "metric_definition",
      "url": "https://www.srtr.org/...",
      "similarity": 0.87
    }
  ],
  "confidence": "high"
}

Document Types and Parsing
1. Metric Definitions (metric_definition)
Source: SRTR metrics glossary pages
Structure:
json[
  {
    "metric_code": "1YR_PAT_SURV",
    "title": "1-year patient survival",
    "definition": "The percentage of patients...",
    "notes": "Risk adjustment methodology..."
  }
]
Chunking: Each metric = 1 chunk
2. Wait Time Calculator (wait_time)
Source: Wait time methodology pages
Structure:
json{
  "title": "Wait Time Calculator",
  "overview": "...",
  "input_variables": [...],
  "methodology": "...",
  "interpretation": "..."
}
Chunking: By major sections (overview, methodology, etc.)
3. Center Pages (center_page)
Source: Individual transplant center PSR pages
Structure:
json{
  "center_name": "UCLA Medical Center",
  "center_id": "UCLA_kidney",
  "location": "Los Angeles, CA",
  "metrics": [...]
}
Chunking: Overview + metrics sections
4. General Pages
Source: About, methodology, OPO info pages
Chunking: By HTML sections, ~400-500 tokens each with 50-token overlap

API Endpoints
POST /api/query
Main query endpoint
Request:
json{
  "prompt": "What is 1-year patient survival?"
}
Response:
json{
  "answer_text": "1-year patient survival represents...",
  "sources": [
    {
      "section": "1-year Patient Survival",
      "doc_type": "metric_definition",
      "url": "https://www.srtr.org/...",
      "similarity": 0.87
    }
  ],
  "confidence": "high",
  "plan": {
    "intent": "metric_definition",
    "filters": {"doc_type": "metric_definition"},
    "rationale": "User wants metric definition"
  }
}
GET /api/debug?q=...
Debug retrieval without full pipeline
Response:
json{
  "query": "kidney transplant survival",
  "retrieved_chunks": 5,
  "chunks": [
    {
      "text": "...",
      "section": "1-year Patient Survival",
      "similarity": 0.87
    }
  ]
}
POST /api/rebuild_index
Rebuild indexes (admin only)

Setup Instructions
1. Prepare HTML Mirrors
bash# Create directory structure
mkdir -p data_repo/html_mirrors
mkdir -p data_repo/meta
Place your HTML files in html_mirrors/:

metrics_definitions.html
wait_time_calculator.html
center_UCLA_kidney.html
etc.

2. Build RAG Index
bash# Install dependencies
pip install beautifulsoup4 openai

# Set API key
export OPENAI_API_KEY="your-key-here"

# Build index
python build_rag_index.py
Expected output:
Building documents index...
  Added: metrics_definitions (type: metric_definition)
  Added: wait_time_calculator (type: wait_time)
Saved 10 documents to index

Building chunks and embeddings...
Processing: metrics_definitions
  Created 25 chunks
    Processed chunk: ch_a1b2c3...
...
Saved 150 chunks to index

RAG Index Build Complete!
3. Start Django Server
bashpython manage.py runserver
4. Test Query
bashcurl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 1-year patient survival?"}'

Advantages of RAG Architecture
vs. Direct File Reading
FeatureOld (File Reading)New (RAG)Data SourceLocal JSON/CSVHTML mirrors (real SRTR content)SearchExact keyword matchSemantic similarityFlexibilityFixed structureAny HTML formatCoverageLimited to curated filesEntire SRTR websiteAccuracyGood for exact matchesGood for fuzzy queries
vs. Direct LLM Knowledge
FeaturePure LLMRAGSourceTraining data (outdated)Live SRTR mirrorsAccuracyMay hallucinateGrounded in actual docsTraceabilityNo citationsFull source attributionCostLowerSlightly higher (embeddings)

Future Enhancements
Phase 1 (Current)

✅ HTML parsing
✅ Semantic chunking
✅ Vector retrieval
✅ Strict citation

Phase 2 (Planned)

 Hybrid search (keyword + semantic)
 Re-ranking for better precision
 Multi-hop reasoning (answer requires multiple documents)
 Caching frequently asked questions

Phase 3 (Advanced)

 Automatic HTML re-crawling
 Change detection and index updates
 Graph-based retrieval (center → metrics → definitions)
 Numeric calculation tools integration


Monitoring and Debugging
Check Index Stats
pythonimport json
from pathlib import Path

docs = json.load(open("data_repo/meta/documents.index.json"))
chunks = json.load(open("data_repo/meta/chunks.index.json"))

print(f"Documents: {len(docs)}")
print(f"Chunks: {len(chunks)}")
print(f"Avg chunks per doc: {len(chunks) / len(docs):.1f}")
Test Retrieval
bashcurl "http://localhost:8000/api/debug?q=kidney%20transplant%20survival"
Check Embedding Quality
Low similarity scores (<0.6) may indicate:

Poor chunk quality (too short/long)
Mismatched document types
Query-document mismatch

Solution: Adjust chunking strategy or add more context to chunks

Troubleshooting
No chunks retrieved
Causes:

Embeddings not generated
Filters too restrictive
Query too vague

Solutions:

Check embeddings/ directory exists and has files
Remove filters in /api/debug to test
Make query more specific

Low quality answers
Causes:

Retrieved chunks not relevant
Chunks too fragmented
LLM ignoring instructions

Solutions:

Increase top_k in retrieval
Adjust chunk overlap
Strengthen Agent B prompt

Slow performance
Causes:

Too many chunks to search
Embedding API calls

Solutions:

Use local embedding model (sentence-transformers)
Pre-filter by document type before similarity search
Cache query embeddings


Security Considerations
✅ Safe:

All data stays local
Only text chunks sent to OpenAI (no patient data)
HTML mirrors don't contain PHI

⚠️ Caution:

Validate HTML sources before indexing
Sanitize user queries to prevent injection
Rate limit API calls


Cost Estimation
One-time indexing:

100 documents → ~500 chunks
Embedding cost: ~$0.01 (at $0.00002/1K tokens)
Time: ~5-10 minutes

Per query:

1 query embedding: ~$0.00001
1 answer generation: ~$0.005-0.02 (depending on context length)

Monthly estimate (1000 queries):

~$5-20 depending on complexity


Maintenance
Regular Tasks
Weekly:

Check for new SRTR pages to mirror
Validate existing mirrors still accessible

Monthly:

Rebuild index if significant SRTR updates
Review retrieval quality metrics
Update HTML parsers if structure changed

Quarterly:

Audit answer quality (sample 100 queries)
Update prompt templates
Optimize chunk boundaries


Summary
The new RAG architecture enables:

Scalability: Handle entire SRTR website, not just curated files
Flexibility: Semantic search, fuzzy matching
Accuracy: Grounded in actual SRTR documentation
Traceability: Full source citations
Maintainability: Easy to update with new content

Key Principle: Agent B can only answer based on retrieved chunks, never fabricate.


Goal
1 data privacy(can't be access by gpt/user)
2 code modification on site
3 record feedback manually（from people）
4 documentation 
5 multi-gent：tasks：1. one generate plot 2. one act as code inspector 3. one written reprot4. one collect good feedback
6 user profile saved

example questions:
I'm 45 years old, type O blood, been on dialysis for 2 years, and my CPRA is 85. How long will I wait?
1. How is kidney function calculated? 
2. What variables are related to donor age?
3. What is DONOR_ID
4. Explain the meaning of eGFR

website base-database（一上来就可以用）的collection（不需要像lambda安装，部署大模型）
（未整合数据，只通过页面获取）case study，broad + basic data module
（整合数据版）用SRTR做case study/application（lambda也没有）

先写manuscript，部署lambda（比较我们的和他的，simulation study）
