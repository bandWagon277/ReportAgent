import json
import logging
import textwrap
from typing import Dict, Any
from gptapp.services.llm import LLMService

logger = logging.getLogger(__name__)

class InitialAgent:
    """
    Initial Agent: 
    """

    def __init__(self, llm_service: LLMService):
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
        self.llm = llm_service

    def plan(self, user_query: str) -> Dict[str, Any]:
        """       
        Returns:
            dict: include intent, answer_mode, entity_identifiers
        """
        system_prompt = textwrap.dedent("""
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
        - "data_dictionary_lookup": User asks about specific variable definitions (e.g., TX_DATE)
        - "concept_explanation": User asks for explanation of a key concept or model (e.g., KDPI, eGFR)
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
        - variable_name: The exact, capitalized variable name (e.g., 'TX_DATE')
        - concept_keywords: Key concept acronyms or names (e.g., ['KDPI', 'eGFR'])
            
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
        "rationale": "User is asking for a specific variable definition."
        }

        Q: "Calculate my kidney waiting time. I'm 30, AB blood type."
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
        """).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            raw_response = self.llm.chat(messages, temperature=0.2, max_tokens=600)
            
            plan = json.loads(raw_response)
            
            plan.setdefault("answer_mode", "textual")
            plan.setdefault("entity_identifiers", {})
            plan.setdefault("semantic_scope", [])
            plan.setdefault("filters", {})
            
            logger.info(f"Initial Agent plan: intent={plan.get('intent')}, "
                   f"answer_mode={plan.get('answer_mode')}, "
                   f"entities={plan.get('entity_identifiers')}")
            return plan

        except Exception as e:
            logger.exception(f"Initial Agent failed to generate plan: {e}")
            
            return {
                "intent": "general",
                "answer_mode": "textual",
                "entity_identifiers": {},
                "semantic_scope": [],
                "filters": {},
                "retrieval_needed": True,
                "use_deterministic_tool": False,
                "rationale": f"Fallback due to error: {str(e)}"
            }