import json
import os
import csv
import logging
import requests
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.core.files.storage import default_storage

# Setup logger
logger = logging.getLogger(__name__)

def render_gpt_interface(request):
    """Render the HTML interface for GPT interaction."""
    return render(request, 'gpt_interface.html')

@csrf_exempt
def call_gpt_api(request):
    """Call the OpenAI GPT API with a user-provided prompt."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_prompt = data.get('prompt', '')

            if not user_prompt:
                return HttpResponse('No prompt provided', status=400)

            payload = {
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "model": "gpt-3.5-turbo",
            }
            api_key = os.getenv('OPENAI_API_KEY', '')
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            }

            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)

            logger.info("Response from OpenAI API: %s", response.text)
            if response.status_code == 200:
                content = response.json()
                generated_text = content.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                # Optionally save the generated code to a file
                save_generated_code("generated_code", generated_text)

                return HttpResponse(generated_text)
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return HttpResponse('Failed to communicate with OpenAI API', status=response.status_code)
        except json.JSONDecodeError:
            return HttpResponse('Invalid JSON', status=400)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to OpenAI API failed: {e}")
            return HttpResponse('Failed to make request to OpenAI API', status=500)
    else:
        return HttpResponse('Method not allowed', status=405)

def save_generated_code(description, generated_text):
    """Save the generated code to a file."""
    if generated_text:
        with open(f'generated_code_{description}.py', 'w') as file:
            file.write(generated_text)
        logger.info(f"Generated code saved to generated_code_{description}.py")
    else:
        logger.error("Empty response from server. Cannot save generated code.")

@csrf_exempt
def upload_csv(request):
    """Upload a CSV file and return its stored path."""
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        if csv_file:
            file_path = default_storage.save(f'uploads/{csv_file.name}', csv_file)
            return JsonResponse({'file_path': file_path}, status=200)
        else:
            return JsonResponse({'error': 'No CSV file provided'}, status=400)
    else:
        return HttpResponse('Method not allowed', status=405)

@csrf_exempt
def execute_code(request):
    """Execute generated code on the uploaded CSV data."""
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        file_path = data.get('file_path', '')
        generated_code = data.get('generated_code', '')

        if not file_path or not generated_code:
            return JsonResponse({'error': 'Missing file path or generated code'}, status=400)

        try:
            full_path = default_storage.path(file_path)
            with open(full_path, 'r') as file:
                csv_reader = csv.reader(file)
                csv_data = list(csv_reader)

            # Execute the generated code
            # WARNING: Executing arbitrary code is dangerous. Use with caution.
            exec_globals = {}
            exec(generated_code, {'__builtins__': {}}, exec_globals)
            processed_data = exec_globals.get('process_csv', lambda x: None)(csv_data)

            return JsonResponse({'processed_data': processed_data}, status=200)
        except Exception as e:
            logger.error(f"Error executing code on CSV data: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return HttpResponse('Method not allowed', status=405)

