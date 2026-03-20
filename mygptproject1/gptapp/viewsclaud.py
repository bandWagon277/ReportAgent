import json
import logging
import os
import requests
import pandas as pd
from io import StringIO
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import subprocess
from django.core.files.storage import FileSystemStorage
from django.core.cache import cache  # 新增：用于临时存储结果
import uuid  # 新增：生成唯一标识符

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
        refine_prompt = "For the following request of producing a csv analysis script，please generate as the order 1.example csv；2.pyspark code；3.python code. In the code, consider we want to analysis a csv data named df (already exist in global environment. It is read in by pd.read_csv,but its other information are unkown;) and final result should name processed_data. No explanation needed.Request:"

        # Call the OpenAI GPT API with the user-provided prompt
        payload = {
            "messages": [
                {"role": "user", "content": refine_prompt + user_prompt}
            ],
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
                generated_text = generated_text.replace('\\', '')
                print(generated_text)
                
                # Process the generated text to create and manipulate CSV
                csv_data, sas_code, python_code = process_prompt(generated_text)

                # Save the CSV data, SAS code, and Python code to fixed file names
                csv_file_path = save_to_file("data.csv", csv_data)
                sas_file_path = save_to_file("code.sas", sas_code)
                py_file_path = save_to_file("code.py", python_code)

                # Execute the Python code on the CSV data
                python_results = execute_python_code(csv_file_path, py_file_path)
                if "Error" in str(python_results):
                    logger.error(f"Execution OpenAI API error: {python_results}")
                    return HttpResponse(str(python_results), status=500)
                else:
                    # 生成唯一的结果ID并缓存结果数据
                    result_id = str(uuid.uuid4())
                    cache.set(f'result_{result_id}', python_results, timeout=3600)  # 缓存1小时
                    
                    # 返回包含结果ID的响应，用于前端预览
                    return JsonResponse({
                        'success': True,
                        'generated_text': generated_text,
                        'result_id': result_id,
                        'preview_data': python_results[:1000] + '...' if len(python_results) > 1000 else python_results  # 预览前1000字符
                    })

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
        else:
            return JsonResponse({'error': 'No CSV file provided'}, status=400)
    else:
        return HttpResponse('Method not allowed', status=405)
    

@csrf_exempt
def process_csv(request):
    """处理CSV文件并返回预览结果"""
    if request.method == 'POST':
        # Get uploaded CSV file and Python script path
        data = json.loads(request.body.decode('utf-8'))
        csv_file_path = data.get('file_path', '')
        
        # Save the uploaded CSV file to a temporary location
        fs = FileSystemStorage()
        script_path = os.path.join(default_storage.location, "code.py")
        
        # Execute the Python script with the CSV file as input
        try:
            # Using subprocess to call the python script with arguments
            python_results = execute_python_code(csv_file_path, script_path)
            
            if "Error" in str(python_results):
                return JsonResponse({'error': str(python_results)}, status=500)
            
            # 生成唯一的结果ID并缓存结果数据
            result_id = str(uuid.uuid4())
            cache.set(f'result_{result_id}', python_results, timeout=3600)  # 缓存1小时
            
            # 返回预览数据和结果ID
            return JsonResponse({
                'success': True,
                'result_id': result_id,
                'preview_data': python_results[:1000] + '...' if len(python_results) > 1000 else python_results  # 预览前1000字符
            })
            
        except subprocess.CalledProcessError as e:
            return JsonResponse({'error': f"Error executing script: {e}"}, status=500)
    
    return render(request, 'gpt_interface.html')


@csrf_exempt
def preview_result(request):
    """预览处理结果"""
    if request.method == 'GET':
        result_id = request.GET.get('result_id')
        if not result_id:
            return JsonResponse({'error': 'No result ID provided'}, status=400)
        
        # 从缓存中获取结果数据
        result_data = cache.get(f'result_{result_id}')
        if not result_data:
            return JsonResponse({'error': 'Result not found or expired'}, status=404)
        
        # 解析CSV数据以获取更好的预览格式
        try:
            df = pd.read_csv(StringIO(result_data))
            
            # 准备预览数据
            preview_info = {
                'success': True,
                'result_id': result_id,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'sample_data': df.head(10).to_dict('records'),  # 前10行数据
                'data_types': df.dtypes.astype(str).to_dict(),
                'file_size': len(result_data)  # 估算文件大小
            }
            
            return JsonResponse(preview_info)
            
        except Exception as e:
            logger.error(f"Error parsing result data for preview: {e}")
            return JsonResponse({
                'success': True,
                'result_id': result_id,
                'raw_preview': result_data[:2000] + '...' if len(result_data) > 2000 else result_data,
                'file_size': len(result_data)
            })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def download_result(request):
    """下载处理结果"""
    if request.method == 'GET':
        result_id = request.GET.get('result_id')
        if not result_id:
            return JsonResponse({'error': 'No result ID provided'}, status=400)
        
        # 从缓存中获取结果数据
        result_data = cache.get(f'result_{result_id}')
        if not result_data:
            return JsonResponse({'error': 'Result not found or expired'}, status=404)
        
        # 创建HTTP响应用于文件下载
        response = HttpResponse(result_data, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="processed_data_{result_id[:8]}.csv"'
        
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
    """Execute the Python code on the CSV data."""
    try:
        # Load the CSV data
        if csv_file_path is None or py_file_path is None:
            logger.error("CSV file or Python file path is None.")
            raise ValueError("CSV file or Python file path is None.")

        with default_storage.open(csv_file_path, 'r') as csv_file:
            csv_data = csv_file.read()

        if not csv_data.strip():
            logger.error("CSV file is empty.")
            raise ValueError("CSV file is empty.")

        df = pd.read_csv(StringIO(csv_data))

        # Cast all columns to string type to avoid FutureWarning
        df = df.astype(str)

        # Load and execute the Python code
        with default_storage.open(py_file_path, 'r') as py_file:
            python_code = py_file.read()

        logger.info(f"Executing Python code: {python_code}")

        exec_globals = {
            'df': df,
            'pd': pd,
            'StringIO': StringIO,
            'default_storage': default_storage,
            'processed_data': None  # Placeholder for processed data
        }
        try:
            print(df)

            exec(python_code, exec_globals)
            # Ensure processed_data is updated
            if 'processed_data' in exec_globals and exec_globals['processed_data'] is not None:
                df = exec_globals['processed_data']
            else:
                logger.warning("Processed data not set in executed Python code. Using original DataFrame.")

        except KeyError as e:
            logger.error(f"KeyError in executed Python code: {e}")
            return f"KeyError: {e}"
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return f"Error: {e}"

        # Collect the results
        results_buffer = StringIO()
        df.to_csv(results_buffer, index=False)
        results_buffer.seek(0)

        return results_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error executing Python code on CSV data: {e}")
        return f"Error: {e}"