# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains multiple Django-based applications that use the OpenAI API (GPT-4o and embeddings) to analyze medical/dialysis data, generate reports, and answer questions about SRTR (Scientific Registry of Transplant Recipients) data.

## Repository Structure

Active projects:

- **`SRTR/`** — The most architecturally advanced project. Multi-agent system for querying SRTR transplant data with RAG retrieval, data dictionary lookup, kidney transplant calculators, and intent-based routing. This is the active development focus.
- **`mygptproject1/`** — CSV/PDF/image processing pipeline. Accepts user-uploaded data, generates Python code via GPT, executes it, and produces reports (DFR-style dialysis facility reports), synthetic data, and visualizations.
  - **`mygptproject1/prompts/`** — All instruction prompt templates (CSV, image, PDF, PDF_A/B variants).
- **`myenv/`** — Python virtual environment.

Other directories:

- **`docs/references/`** — Reference PDFs (LAMBDA.pdf, kidney/mortality papers).
- **`archive/`** — Superseded projects (`sythetic_data/`).
- **`django/`** — Cloned Django framework source (not a project app).

## Running the Projects

```bash
# Activate the virtual environment
source myenv/bin/activate  # Linux/WSL
# or: myenv\Scripts\activate  # Windows

# Run the SRTR project
cd SRTR && python manage.py runserver

# Run the mygptproject1 project
cd mygptproject1 && python manage.py runserver
```

All projects require the `OPENAI_API_KEY` environment variable to be set. The models default to `gpt-4o` (chat) and `text-embedding-3-small` (embeddings), configurable via `OPENAI_MODEL` and `EMBEDDING_MODEL` env vars.

## SRTR Architecture (Multi-Agent System)

The SRTR project uses a layered agent architecture:

1. **`views.py`** — Django request handlers. Entry point is `api_query` at `/api/query`.
2. **`agents/planner.py` (InitialAgent)** — Classifies user intent (calculator, data_dictionary_lookup, concept_explanation, or general RAG) and determines answer mode (numeric vs textual).
3. **`agents/orchestrator.py` (MainAgent)** — Routes to appropriate handler based on the plan.
4. **`services/llm.py` (LLMService)** — Wrapper around OpenAI Chat and Embedding APIs using raw `requests`.
5. **`services/rag.py` (RAGEngine)** — Two-stage retrieval: keyword pre-filtering then cosine similarity on stored embeddings.
6. **`services/data_manager.py` (DataManager)** — Reads/indexes CSV data dictionaries under `data_repo/dictionaries/`.
7. **`tools/kidney.py`** — Kidney transplant waiting time calculator.

Data lives in `SRTR/data_repo/`: dictionaries (CSV), concepts (R code/txt), and a JSON index at `data_repo/meta/dictionaries.index.json`.

Key API endpoints (SRTR):
- `GET /gpt-interface/` — Main web UI
- `POST /api/query` — Process a user query through the agent pipeline
- `POST /api/rebuild_index` — Rebuild the data dictionary and document indices

## mygptproject1 Architecture (Report Generation)

Uses a two-agent pipeline:
- **Agent A** — Receives user data (CSV/PDF/image) + instruction prompt, generates Python analysis code via GPT.
- **Agent B** — Takes Agent A's code output and composes it into a formatted report with figures/tables.

Key endpoints: `/process-csv/`, `/upload_files/`, `/execute_pdf_pipeline/`, `/synthetic/`.

Artifacts (generated figures, tables, code) are saved to `media/` with UUID-based subdirectories.

## Key Conventions

- All LLM calls go through OpenAI-compatible REST API (not the official Python SDK) using `requests.post`.
- `MEDIA_ROOT` is hardcoded to `C:/Users/18120/Desktop/OPENAIproj/media` in settings files — update if working from a different path.
- Comments and variable names mix English and Chinese (中文注释). Both languages appear in code comments, settings, and architecture docs.
- Instruction prompts for different data types are stored in `mygptproject1/prompts/` (`Instruction_prompt_pdf.txt`, `Instruction_prompt_csv.txt`, etc.). The path is resolved via `get_prompt_path()` / `get_pdf_dual_prompts()` in `gpt_backend_utils.py`, defaulting to `<project_root>/prompts/` (overridable with `PROMPT_BASE_PATH` env var).
- The `gpt_backend_utils.py` files contain shared utility functions for code execution, LLM interaction, and artifact management.
- Templates are in `gptapp/templates/` within each project.
