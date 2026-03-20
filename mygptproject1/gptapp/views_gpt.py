import json
import logging
import os
import requests
import pandas as pd
import base64  # Added for encoding binary data for previews
from io import StringIO
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile  # Ensure this import is present
import subprocess
from django.core.files.storage import FileSystemStorage
from django.core.cache import cache  # 新增：用于临时存储结果
import uuid

# Define logger
logger = logging.getLogger(__name__)


@csrf_exempt
def render_gpt_interface(request):
    """Render the HTML interface for GPT interaction and handle form submission."""
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        user_prompt = data.get('prompt', '')
        if not user_prompt:
            return HttpResponse('No prompt provided', status=400)
        
        #add instruction
        prompt_path = r"C:/Users/18120/Desktop/OPENAIproj/Instruction_prompt.txt"

        with open(prompt_path, "r", encoding="utf-8") as f:
            refine_prompt = f.read().strip()

        # Call the OpenAI GPT API with the user-provided prompt
        payload = {
            "messages": [
                {"role": "user", "content": refine_prompt + user_prompt}
            ],
            #"model": "gpt-3.5-turbo",
            "model": "gpt-4o",
        }
        api_key = os.getenv('OPENAI_API_KEY', '')
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
            logger.info("Response from OpenAI API: %s", response.text)
            if response.status_code == 200:
                content = response.json()
                generated_text = content.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                # Clean up generated text by removing unwanted escape sequences or backslashes
                generated_text = generated_text.replace('\\', '')  # Remove backslashes if not necessary
                print(generated_text)
                # Process the generated text to create and manipulate CSV
                csv_data, sas_code, python_code = process_prompt(generated_text)

                # Save the CSV data, SAS code, and Python code to fixed file names
                csv_file_path = save_to_file("data.csv", csv_data)
                sas_file_path = save_to_file("code.sas", sas_code)
                py_file_path = save_to_file("code.py", python_code)

                # Execute the Python code on the CSV data to validate it runs without fatal errors.
                python_results = execute_python_code(csv_file_path, py_file_path)
                # If an error occurred during execution, return a 500 with the error message
                if isinstance(python_results, dict) and python_results.get('error'):
                    err_msg = python_results['error']
                    logger.error(f"Execution OpenAI API error: {err_msg}")
                    return HttpResponse(err_msg, status=500)
                # Otherwise return the raw generated text. The actual execution result
                # will be processed in the process_csv endpoint.
                return HttpResponse(generated_text, content_type='text/plain')
                """return render(request, 'gpt_interface.html', {
                #'csv_file_path': csv_file_path,
                'script_path': py_file_path,
                'result_message': generated_text,
            })"""

            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return HttpResponse('Failed to communicate with OpenAI API', status=response.status_code)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to OpenAI API failed: {e}")
            return HttpResponse('Failed to make request to OpenAI API', status=500)
    else:
        return render(request, 'gpt_interface.html')




@csrf_exempt
def upload_csv(request):
    """Upload a CSV file and return its stored path."""
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        if csv_file:
            file_path = default_storage.save(f'uploads/{csv_file.name}', csv_file)
            return JsonResponse({'file_path': file_path}, status=200)
            """return render(request, 'gpt_interface.html', {
                'csv_file_path': file_path,
                'result_message': 'file_path:'+file_path,
            })"""
        else:
            return JsonResponse({'error': 'No CSV file provided'}, status=400)
    else:
        return HttpResponse('Method not allowed', status=405)
    

@csrf_exempt
def process_csv(request):
    if request.method == 'POST':
        # Get uploaded CSV file and Python script path
        data = json.loads(request.body.decode('utf-8'))
        csv_file_path =  data.get('file_path', '')
        
        # Save the uploaded CSV file to a temporary location
        fs = FileSystemStorage()
        
        # Define the output file path for the results: result_file_path = fs.path('result.txt')

        script_path = os.path.join(default_storage.location, "code.py")
        
        # Execute the Python script with the CSV file as input
        try:

            # Execute the user-provided script
            python_results = execute_python_code(csv_file_path, script_path)
            # Handle errors returned from execution
            if isinstance(python_results, dict) and python_results.get('error'):
                return JsonResponse({'error': python_results['error']}, status=500)

            # At this point python_results should be a dict with keys: data, type, file_name
            if not isinstance(python_results, dict) or 'type' not in python_results:
                return JsonResponse({'error': 'Invalid result from execution'}, status=500)

            result_type = python_results.get('type')
            result_data = python_results.get('data')

            # Generate a unique identifier for caching the result
            result_id = str(uuid.uuid4())
            cache.set(f'result_{result_id}', python_results, timeout=3600)

            # Prepare a small preview for CSV results only (first 1000 characters)
            preview_data = None
            if result_type == 'csv' and isinstance(result_data, str):
                preview_data = result_data[:1000] + '...' if len(result_data) > 1000 else result_data

            return JsonResponse({
                'success': True,
                'result_id': result_id,
                'file_type': result_type,
                'preview_data': preview_data
            })
            
        except subprocess.CalledProcessError as e:
            return HttpResponse(f"Error executing script: {e}")
        
        # After processing, pass the path of the result file back to the template for download
        # return JsonResponse({'result_path': python_results}, status=200)
    
    return render(request, 'gpt_interface.html')

@csrf_exempt
def preview_result(request):
    """预览处理结果"""
    if request.method == 'GET':
        result_id = request.GET.get('result_id')
        if not result_id:
            return JsonResponse({'error': 'No result ID provided'}, status=400)
        
        # 从缓存中获取结果数据
        cached_result = cache.get(f'result_{result_id}')
        if not cached_result:
            return JsonResponse({'error': 'Result not found or expired'}, status=404)

        # Expect cached_result to be a dict with keys: data, type, file_name
        result_type = cached_result.get('type')
        result_data = cached_result.get('data')

        # CSV preview handling
        if result_type == 'csv' and isinstance(result_data, str):
            try:
                df = pd.read_csv(StringIO(result_data))
                preview_info = {
                    'success': True,
                    'result_id': result_id,
                    'file_type': 'csv',
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'columns': list(df.columns),
                    'sample_data': df.head(10).fillna('').to_dict('records'),
                    'data_types': df.dtypes.astype(str).to_dict(),
                    'file_size': len(result_data)
                }
                return JsonResponse(preview_info)
            except Exception as e:
                logger.error(f"Error parsing CSV result data for preview: {e}")
                truncated = result_data[:2000] + '...' if len(result_data) > 2000 else result_data
                return JsonResponse({
                    'success': True,
                    'result_id': result_id,
                    'file_type': 'csv',
                    'raw_preview': truncated,
                    'file_size': len(result_data)
                })

        # Image preview handling
        if result_type == 'image' and isinstance(result_data, (bytes, bytearray)):
            try:
                # Determine image MIME type based on file extension
                file_name = cached_result.get('file_name', '')
                mime = 'image/png'
                if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                    mime = 'image/jpeg'
                # Encode image to base64 for preview
                b64 = base64.b64encode(result_data).decode('utf-8')
                preview_info = {
                    'success': True,
                    'result_id': result_id,
                    'file_type': 'image',
                    'file_size': len(result_data),
                    'preview_image': f"data:{mime};base64,{b64}"
                }
                return JsonResponse(preview_info)
            except Exception as e:
                logger.error(f"Error preparing image preview: {e}")
                return JsonResponse({
                    'success': True,
                    'result_id': result_id,
                    'file_type': 'image',
                    'file_size': len(result_data),
                    'preview_image': None
                })

        # PDF preview handling: no preview, just return file size
        if result_type == 'pdf' and isinstance(result_data, (bytes, bytearray)):
            return JsonResponse({
                'success': True,
                'result_id': result_id,
                'file_type': 'pdf',
                'file_size': len(result_data)
            })

        # Fallback if type is unknown
        return JsonResponse({
            'error': 'Unsupported result type for preview',
            'file_type': result_type or 'unknown'
        }, status=400)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def download_result(request):
    """下载处理结果"""
    if request.method == 'GET':
        result_id = request.GET.get('result_id')
        if not result_id:
            return JsonResponse({'error': 'No result ID provided'}, status=400)
        
        # 从缓存中获取结果数据
        cached_result = cache.get(f'result_{result_id}')
        if not cached_result:
            return JsonResponse({'error': 'Result not found or expired'}, status=404)

        # Expect cached_result to be a dict with keys: data, type, file_name
        result_data = cached_result.get('data')
        result_type = cached_result.get('type')
        file_name = cached_result.get('file_name')

        # Determine appropriate content type and default filename based on result type
        if result_type == 'csv' and isinstance(result_data, str):
            content_type = 'text/csv'
            download_name = file_name or f"processed_data_{result_id[:8]}.csv"
            response = HttpResponse(result_data, content_type=content_type)
            response['Content-Disposition'] = f'attachment; filename="{download_name}"'
        elif result_type == 'image' and isinstance(result_data, (bytes, bytearray)):
            # Determine MIME type based on file extension
            mime = 'image/png'
            if file_name and (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg')):
                mime = 'image/jpeg'
            download_name = file_name or f"generated_image_{result_id[:8]}.png"
            response = HttpResponse(result_data, content_type=mime)
            response['Content-Disposition'] = f'attachment; filename="{download_name}"'
        elif result_type == 'pdf' and isinstance(result_data, (bytes, bytearray)):
            content_type = 'application/pdf'
            download_name = file_name or f"report_{result_id[:8]}.pdf"
            response = HttpResponse(result_data, content_type=content_type)
            response['Content-Disposition'] = f'attachment; filename="{download_name}"'
        else:
            return JsonResponse({'error': 'Unsupported result type for download', 'file_type': result_type or 'unknown'}, status=400)

        logger.info(f"Result {result_id} downloaded successfully")
        return response
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def process_prompt(generated_text):
    """Process the prompt to generate CSV data and corresponding code snippets."""
    csv_data = ''
    sas_code = ''
    python_code = ''

    # Splitting the generated text by the markers and identifying sections
    sections = generated_text.split('```')
    for i, section in enumerate(sections):
        section = section.strip()
        if i == 1:
            csv_data = section
        elif i == 3 and 'sas' in sections[i-1].lower():
            sas_code_lines = section.split('\n')
            sas_code = '\n'.join(sas_code_lines[1:]).strip()  # Skip the first line
        elif i == 5 and 'python' in sections[i-1].lower():
            python_code_lines = section.split('\n')
            python_code = '\n'.join(python_code_lines[1:]).strip()  # Skip the first line
    """csv_find = False
    sas_find = False
    python_find = False
    for i, section in enumerate(sections):
        section = section.strip()

        #if 'csv' in sections[i-1].lower() and 'id' in sections[i-1].lower() and not csv_find:
        if 'csv' in sections[i-1].lower() and i==1 and not csv_find:
            csv_data = section
            csv_find = True
        if 'sas' in sections[i-1].lower() and ';' in sections[i-1].lower() and not sas_find:
            sas_code_lines = section.split('\n')
            sas_code = '\n'.join(sas_code_lines[1:]).strip()  # Skip the first line
            sas_find = True
        if 'python' in sections[i-1].lower() and ('def' in sections[i-1].lower() or 'import' or "#") and not python_find:
            python_code_lines = section.split('\n')
            python_code = '\n'.join(python_code_lines[1:]).strip()  # Skip the first line
            python_find = True"""

    logger.info(f"CSV data: {csv_data}")
    logger.info(f"SAS code: {sas_code}")
    logger.info(f"Python code: {python_code}")

    return csv_data, sas_code, python_code


def save_to_file(file_name, content):
    """Save the given content to a file with the specified name."""
    if not content.strip():
        logger.warning(f"No content to save for {file_name}")
        return None

    # Define the full path for the file
    file_path = os.path.join(default_storage.location, file_name)

    # Save the content directly to the specified file name
    with default_storage.open(file_path, 'w') as f:
        f.write(content)

    if default_storage.exists(file_path):
        logger.info(f"Saved content to {file_path}")
    else:
        logger.error(f"Failed to save content to {file_name}")

    return file_path

def execute_python_code(csv_file_path, py_file_path):
    """
    Execute the Python code on the CSV data.

    This helper inspects the executed code's namespace to determine which type of
    result was produced. It supports three output types:

    * CSV: if a DataFrame is assigned to ``processed_data``. The DataFrame will
      be converted back to a CSV string.
    * Image: if ``image_path`` is defined and points to an existing file in
      ``default_storage``. The file bytes are returned along with the filename.
    * PDF: if ``pdf_path`` is defined and points to an existing file in
      ``default_storage``. The file bytes are returned along with the filename.

    The function returns a dictionary with keys ``data``, ``type``, and
    ``file_name``. On error, a dictionary with an ``error`` key is returned.
    """
    try:
        # Validate inputs
        if csv_file_path is None or py_file_path is None:
            logger.error("CSV file or Python file path is None.")
            return {'error': 'CSV file or Python file path is None.'}

        # Read CSV content
        with default_storage.open(csv_file_path, 'r') as csv_file:
            csv_data = csv_file.read()

        if not csv_data or not csv_data.strip():
            logger.error("CSV file is empty.")
            return {'error': 'CSV file is empty.'}

        # Load DataFrame and cast to string to avoid dtype issues
        df = pd.read_csv(StringIO(csv_data))
        df = df.astype(str)

        # Read Python code
        with default_storage.open(py_file_path, 'r') as py_file:
            python_code = py_file.read()
        logger.info(f"Executing Python code: {python_code}")

        # Prepare execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'StringIO': StringIO,
            'default_storage': default_storage,
            'processed_data': None,
            'image_path': None,
            'pdf_path': None,
        }

        # Execute the user-provided code
        try:
            exec(python_code, exec_globals)
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return {'error': str(e)}

        # Inspect execution results
        # 1. Image output: check if image_path was set and file exists
        image_path = exec_globals.get('image_path')
        if image_path:
            try:
                # Resolve relative paths within default_storage if necessary
                file_path = image_path
                if not default_storage.exists(file_path):
                    # Try to resolve as relative to storage root
                    file_path = os.path.join(default_storage.location, image_path)
                if default_storage.exists(file_path):
                    with default_storage.open(file_path, 'rb') as f:
                        data_bytes = f.read()
                    return {
                        'data': data_bytes,
                        'type': 'image',
                        'file_name': os.path.basename(file_path) or 'generated_image.png'
                    }
            except Exception as e:
                logger.error(f"Error reading generated image: {e}")
                return {'error': f"Error reading generated image: {e}"}

        # 2. PDF output: check if pdf_path was set and file exists
        pdf_path = exec_globals.get('pdf_path')
        if pdf_path:
            try:
                file_path = pdf_path
                if not default_storage.exists(file_path):
                    file_path = os.path.join(default_storage.location, pdf_path)
                if default_storage.exists(file_path):
                    with default_storage.open(file_path, 'rb') as f:
                        data_bytes = f.read()
                    return {
                        'data': data_bytes,
                        'type': 'pdf',
                        'file_name': os.path.basename(file_path) or 'report.pdf'
                    }
            except Exception as e:
                logger.error(f"Error reading generated PDF: {e}")
                return {'error': f"Error reading generated PDF: {e}"}

        # 3. CSV output: check if processed_data is set as DataFrame
        processed_df = exec_globals.get('processed_data')
        if processed_df is not None and isinstance(processed_df, pd.DataFrame):
            df = processed_df
        else:
            logger.warning(
                "Processed data not set in executed Python code. Using original DataFrame."
            )

        # Convert DataFrame to CSV string
        results_buffer = StringIO()
        df.to_csv(results_buffer, index=False)
        results_buffer.seek(0)
        return {
            'data': results_buffer.getvalue(),
            'type': 'csv',
            'file_name': 'processed_data.csv'
        }

    except Exception as e:
        logger.error(f"Error executing Python code on CSV data: {e}")
        return {'error': str(e)}
