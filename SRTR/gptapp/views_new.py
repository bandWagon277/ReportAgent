import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from gptapp.services.llm import LLMService
from gptapp.agents.orchestrator import MainAgent
#from gptapp.tools.kidney import calculate_kidney_waiting_time

logger = logging.getLogger(__name__)

# Inilialize
llm_service = LLMService()
main_agent = MainAgent(llm_service)

# ==========
# 1. 页面渲染 (Page Rendering)
# ==========
def render_gpt_interface(request):
    """Reture HTML interface"""
    return render(request, "gpt_interface.html")

# ==========
# 2. Core API interface (The Traffic Controller)
# ==========
@csrf_exempt
def api_query(request):
    """
    Main query interface:
    1. Check HTTP method
    2. Parse JSON
    3. Distribute the task to the Agent
    4. Package and send back the Agent's results
    """
    # HTTP Protocol Inspection
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        # Input parsing
        data = json.loads(request.body.decode("utf-8"))
        user_prompt = data.get("prompt", "").strip()
        
        if not user_prompt:
            return JsonResponse({"error": "No prompt provided"}, status=400)

        # distribute tasks
        result = main_agent.process_query(user_prompt)

        # Response Formatting
        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("Unexpected error in api_query")
        # Handling Errors as a Catch-Up
        return JsonResponse({
            "error": "Internal Server Error",
            "detail": str(e)
        }, status=500)