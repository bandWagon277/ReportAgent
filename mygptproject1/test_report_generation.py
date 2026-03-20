"""
Test script for PDF report generation pipeline.
Three-agent flow: Agent 0 (Planner) -> Agent A (code gen) -> execute -> Agent B (compose PDF).

Usage:
    cd mygptproject1
    python test_report_generation.py

Requires OPENAI_API_KEY environment variable.
Output: ../srtr_test_report.pdf
"""
import os
import sys
import json
import logging

# ---- Django bootstrap ----
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mygptproject.settings")

import django
django.setup()

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.cache import cache

# ---- project imports ----
from gptapp.gpt_backend_utils import (
    summarize_csv_privacy,
    get_pdf_dual_prompts,
    get_pdf_planner_prompt,
    generate_report_plan,
    generate_agent_a_code,
    extract_python_code,
    save_to_file,
    execute_python_code,
    persist_artifacts_and_build_manifest,
    build_enhanced_agentB_messages,
    call_openai_chat,
    execute_composer_code,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("test_report")

# ---- config ----
CSV_SOURCE = os.path.join(os.path.dirname(__file__), "upload_csv", "srtr_simulation_500.csv")
OUTPUT_PDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "srtr_test_report.pdf"))
USER_PROMPT = (
    "Generate an SRTR-style kidney transplant outcomes report. "
    "Include: study population demographics table (Table 1), "
    "outcome disposition summary, mortality/death rate analysis by subgroup, "
    "donor quality analysis (KDPI distribution), and waitlist duration trends."
)

def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    # ---- Step 0: Upload CSV to Django storage ----
    logger.info("Step 0: Uploading CSV to Django storage...")
    with open(CSV_SOURCE, "r", encoding="utf-8") as f:
        csv_content = f.read()
    csv_storage_path = default_storage.save("uploads/srtr_simulation_500.csv",
                                            ContentFile(csv_content.encode("utf-8")))
    logger.info(f"  CSV stored at: {csv_storage_path}")

    # ---- Step 1: Build privacy summary ----
    logger.info("Step 1: Building privacy-preserving data summary...")
    privacy_summary = summarize_csv_privacy(csv_storage_path)
    logger.info(f"  Schema: {len(privacy_summary['schema'])} cols, {privacy_summary['n_rows']} rows")

    # ---- Step 2: Load prompt templates ----
    logger.info("Step 2: Loading prompt templates (Planner/A/B)...")
    tmpl_planner = get_pdf_planner_prompt()
    tmpl_A, tmpl_B = get_pdf_dual_prompts()
    logger.info(f"  Planner: {len(tmpl_planner)}, Template A: {len(tmpl_A)}, Template B: {len(tmpl_B)}")

    # ---- Step 3: Agent 0 (Planner) - Structure the report ----
    logger.info("Step 3: Calling Agent 0 (Planner) to structure the report...")
    report_plan = generate_report_plan(
        user_prompt=USER_PROMPT,
        privacy_summary=privacy_summary,
        planner_template=tmpl_planner,
        api_key=api_key,
    )
    plan_sections = report_plan.get("sections", [])
    logger.info(f"  Report plan: '{report_plan.get('report_title', 'N/A')}' "
                f"with {len(plan_sections)} sections, "
                f"{report_plan.get('figure_count', '?')} figures, "
                f"{report_plan.get('table_count', '?')} tables")
    logger.info(f"  Planned sections: {[s.get('title', '') for s in plan_sections]}")

    # Save plan for inspection
    plan_path = save_to_file("report_plan.json", json.dumps(report_plan, indent=2, ensure_ascii=False))
    logger.info(f"  Saved report plan to: {plan_path}")

    # ---- Step 4: Agent A - Generate analysis code (guided by plan) ----
    logger.info("Step 4: Calling Agent A (code generation via OpenAI)...")
    raw_agent_a = generate_agent_a_code(
        user_prompt=USER_PROMPT,
        privacy_summary=privacy_summary,
        instruction_template_A=tmpl_A,
        api_key=api_key,
        report_plan=report_plan,
    )
    agent_a_code = extract_python_code(raw_agent_a)
    if not agent_a_code:
        logger.error("Agent A returned no Python code. Raw response preview:")
        print(raw_agent_a[:2000])
        sys.exit(1)
    logger.info(f"  Agent A code extracted ({len(agent_a_code)} chars)")

    codeA_path = save_to_file("agentA_code.py", agent_a_code)
    logger.info(f"  Saved Agent A code to: {codeA_path}")

    # ---- Step 5: Execute Agent A code on real DataFrame ----
    logger.info("Step 5: Executing Agent A code on DataFrame...")
    artifacts = execute_python_code(
        csv_file_path=csv_storage_path,
        py_file_path=codeA_path,
        output_type="analysis",
        csv_path=csv_storage_path,
        image_path=None,
        dry_run=False,
    )
    if isinstance(artifacts, Exception):
        logger.error(f"Agent A execution failed: {artifacts}")
        sys.exit(1)
    if not isinstance(artifacts, dict):
        logger.error(f"Agent A did not produce artifacts dict. Got: {type(artifacts)}")
        sys.exit(1)

    n_fig = len(artifacts.get("figures", []))
    n_tbl = len(artifacts.get("tables", []))
    sections = artifacts.get("sections_implemented", [])
    warnings = artifacts.get("warnings", [])
    logger.info(f"  Artifacts: {n_fig} figures, {n_tbl} tables, sections={sections}")
    if warnings:
        logger.warning(f"  Warnings from Agent A: {warnings}")

    # ---- Step 6: Persist artifacts -> manifest ----
    logger.info("Step 6: Persisting artifacts and building manifest...")
    manifest_id, manifest = persist_artifacts_and_build_manifest(artifacts)
    logger.info(f"  Manifest ID: {manifest_id}")

    # ---- Step 7: Agent B - Compose PDF (with full context) ----
    logger.info("Step 7: Calling Agent B (PDF composition via OpenAI)...")
    messages_B = build_enhanced_agentB_messages(
        manifest=manifest,
        instruction_template_B=tmpl_B,
        user_prompt=USER_PROMPT,
        report_plan=report_plan,
    )
    resp_B = call_openai_chat(api_key, messages_B, timeout=180)
    code_B = extract_python_code(resp_B)
    if not code_B:
        logger.error("Agent B returned no Python code. Raw response preview:")
        print(resp_B[:2000])
        sys.exit(1)
    logger.info(f"  Agent B code extracted ({len(code_B)} chars)")

    codeB_path = save_to_file("agentB_composer.py", code_B)
    logger.info(f"  Saved Agent B code to: {codeB_path}")

    # ---- Step 8: Execute composer to generate PDF ----
    logger.info("Step 8: Executing Agent B composer code...")
    manifest_paths = cache.get(f"manifest:{manifest_id}:paths") or {"figures": {}, "tables": {}}
    pdf_bytes, narrative_text = execute_composer_code(codeB_path, manifest, manifest_paths)

    logger.info(f"  PDF size: {len(pdf_bytes):,} bytes")
    logger.info(f"  Narrative: {narrative_text[:200]}..." if len(narrative_text) > 200 else f"  Narrative: {narrative_text}")

    # ---- Step 9: Save PDF to root directory ----
    with open(OUTPUT_PDF, "wb") as f:
        f.write(pdf_bytes)
    logger.info(f"Step 9: PDF saved to: {OUTPUT_PDF}")
    print(f"\nSUCCESS: Report generated at {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
