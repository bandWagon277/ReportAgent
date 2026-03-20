import json
import logging
import textwrap
from typing import Dict, Any, List

# 1. 引入其他模块
from gptapp.services.llm import LLMService
# 假设你已经把检索逻辑移到了 services/rag.py
from gptapp.services.rag import RAGEngine
# 假设你把数据字典读取逻辑移到了 services/data.py
from gptapp.services.data_manager import DataManager
# 假设你把计算器逻辑移到了 tools/kidney.py
from gptapp.tools.kidney import calculate_kidney_waiting_time, extract_calculator_parameters

from .planner import InitialAgent

logger = logging.getLogger(__name__)

class MainAgent:
    """
    Main Agent (Orchestrator):
    The overall conductor of the system. It receives user requests, calls the InitialAgent for planning,
    then distributes the tasks to the RAG, data dictionary, or calculator tools based on the planning results,
    and finally integrates the results before returning them.
    """

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        
        self.planner = InitialAgent(llm_service)
        self.rag = RAGEngine(llm_service) # require LLM to do embediing
        self.data_manager = DataManager() # read CSV/JSON index
        
    def process_query(self, user_query: str) -> Dict[str, Any]: #combine api_query and _agent_b_generate_answer
        """
        Process user query
        """
        # --- Step 1: get intension and plan ---
        plan = self.planner.plan(user_query)
        intent = plan.get("intent", "general")
        answer_mode = plan.get("answer_mode", "textual")

        logger.info(f"MainAgent processing intent: {intent}, mode: {answer_mode}")

        # --- Step 2: task distributed ---
        try:
            # A: Calculator Tools
            if intent == "calculator" or answer_mode == "calculator":
                return self._handle_calculator(user_query, plan)
            
            # B: Data dictionary lookup
            elif intent == "data_dictionary_lookup":
                return self._handle_dictionary(user_query, plan)
            
            # C: Explanation of Concepts (Local Concept Lookup)
            elif intent == "concept_explanation":
                return self._handle_concept(user_query, plan)
            
            # D: Default RAG retrieval process
            else:
                return self._handle_rag_retrieval(user_query, plan)

        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            return self._generate_error_response(str(e))

    # ==========================
    # Handlers: Process with different tasks
    # ==========================

    def _handle_calculator_query(self,query: str, plan: dict) -> dict:
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
        params = extract_calculator_parameters(query, tool_name)
        
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
                "detail": self.generate_parameter_prompt(tool_name, missing, params),
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
            return self.format_calculator_result(result, query)
        
        else:
            return {
                "summary": f"Calculator '{tool_name}' is not implemented yet.",
                "detail": "This tool is planned but not yet available.",
                "key_points": [],
                "sources": [],
                "confidence": "not_implemented"
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
    
    
        
    # generators

    def generate_parameter_prompt(self, tool_name: str, missing: List[str], partial: dict) -> str:
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

    def format_calculator_result(self, result: dict, original_query: str) -> dict:
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