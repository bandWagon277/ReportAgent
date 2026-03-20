import os
import json
import logging
import requests
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

# Setup logger
logger = logging.getLogger(__name__)

def render_gpt_interface(request):
    return render(request, 'gpt_interface.html')

@csrf_exempt
def call_gpt_api(request):
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

                # Save the generated code to a file
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
    if generated_text:
        # Write the generated text to a file
        with open(f'generated_code_{description}.py', 'w') as file:
            file.write(generated_text)
        logger.info(f"Generated code saved to generated_code_{description}.py")
    else:
        logger.error("Empty response from server. Cannot save generated code.")

@csrf_exempt
def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        if csv_file:
            # Handle the uploaded CSV file
            csv_data = handle_uploaded_csv(csv_file)
            return JsonResponse({'csv_data': csv_data})
        else:
            return JsonResponse({'error': 'No CSV file provided'}, status=400)
    else:
        return HttpResponse('Method not allowed', status=405)

def handle_uploaded_csv(csv_file):
    # Assuming CSV file is properly formatted
    csv_data = []
    try:
        decoded_file = csv_file.read().decode('utf-8').splitlines()
        csv_reader = csv.reader(decoded_file)
        for row in csv_reader:
            csv_data.append(row)
    except Exception as e:
        logger.error(f"Error handling CSV file: {e}")
    return csv_data

@csrf_exempt
def apply_gpt_code(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        prompt = data.get('prompt', '')
        csv_data = data.get('csv_data', [])

        if not prompt:
            return JsonResponse({'error': 'No prompt provided'}, status=400)
        
        if not csv_data:
            return JsonResponse({'error': 'No CSV data provided'}, status=400)

        # Apply GPT code to the CSV data
        gpt_result = apply_gpt_to_csv(prompt, csv_data)
        return JsonResponse({'gpt_result': gpt_result})
    else:
        return HttpResponse('Method not allowed', status=405)

def apply_gpt_to_csv(prompt, csv_data):
    # Prepare the data for input to the GPT model
    formatted_data = '\n'.join(','.join(row) for row in csv_data)

    # Send the data to the GPT API
    api_key = os.getenv('OPENAI_API_KEY')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload = {
        'prompt': prompt,
        'data': formatted_data
    }
    try:
        response = requests.post('https://api.openai.com/v1/apply-gpt', headers=headers, json=payload)
        response.raise_for_status()
        generated_text = response.json().get('generated_text')
        return generated_text
    except Exception as e:
        logger.error(f"Error applying GPT code: {e}")
        return None
