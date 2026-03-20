import json
import logging
import os
import requests
import pandas as pd
from io import StringIO, BytesIO
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import subprocess
from django.core.files.storage import FileSystemStorage
from django.core.cache import cache
import uuid
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from PIL import Image
import tempfile

# Define logger
logger = logging.getLogger(__name__)

# Define output types
OUTPUT_TYPES = {
    'CSV': 'csv',
    'IMAGE': 'image', 
    'PDF_REPORT': 'pdf_report'
}

@csrf_exempt
def render_gpt_interface(request):
    """Render the HTML interface for GPT interaction and handle form submission."""
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        user_prompt = data.get('prompt', '')
        output_type = data.get('output_type', 'CSV')  # New parameter for output type
        
        if not user_prompt:
            return HttpResponse('No prompt provided', status=400)
        
        if output_type not in OUTPUT_TYPES:
            return HttpResponse('Invalid output type', status=400)
        
        # Add instruction based on output type
        prompt_path = get_prompt_path(output_type)
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            refine_prompt = f.read().strip()

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
                generated_text = generated_text.replace('\\', '')
                
                # Process based on output type
                if output_type == 'CSV':
                    return process_csv_output(generated_text)
                elif output_type == 'IMAGE':
                    return process_image_output(generated_text)
                elif output_type == 'PDF_REPORT':
                    return process_pdf_output(generated_text)
                
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return HttpResponse('Failed to communicate with OpenAI API', status=response.status_code)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to OpenAI API failed: {e}")
            return HttpResponse('Failed to make request to OpenAI API', status=500)
    else:
        return render(request, 'gpt_interface.html')

def get_prompt_path(output_type):
    """Get the appropriate prompt file based on output type."""
    base_path = r"C:/Users/18120/Desktop/OPENAIproj/"
    
    if output_type == 'CSV':
        return os.path.join(base_path, "Instruction_prompt_csv.txt")
    elif output_type == 'IMAGE':
        return os.path.join(base_path, "Instruction_prompt_image.txt")
    elif output_type == 'PDF_REPORT':
        return os.path.join(base_path, "Instruction_prompt_pdf.txt")
    else:
        return os.path.join(base_path, "Instruction_prompt.txt")

def process_csv_output(generated_text):
    """Process GPT output for CSV generation."""
    csv_data, sas_code, python_code = process_prompt(generated_text)
    
    # Save files
    csv_file_path = save_to_file("data.csv", csv_data)
    sas_file_path = save_to_file("code.sas", sas_code)
    py_file_path = save_to_file("code.py", python_code)
    
    # Execute Python code
    python_results = execute_python_code(csv_file_path, py_file_path, 'csv')
    
    if isinstance(python_results, Exception):
        logger.error(f"Execution error: {python_results}")
        return HttpResponse(str(python_results), status=500)
    else:
        return HttpResponse(generated_text, content_type='text/plain')

def process_image_output(generated_text):
    """Process GPT output for image generation."""
    csv_data, python_code = process_image_prompt(generated_text)
    
    # Save files
    csv_file_path = save_to_file("data.csv", csv_data)
    py_file_path = save_to_file("image_code.py", python_code)
    
    # Execute Python code for image generation
    result = execute_python_code(csv_file_path, py_file_path, 'image')
    
    if isinstance(result, Exception):
        logger.error(f"Image generation error: {result}")
        return HttpResponse(str(result), status=500)
    else:
        return HttpResponse(generated_text, content_type='text/plain')

def process_pdf_output(generated_text):
    """Process GPT output for PDF report generation."""
    csv_data, python_code = process_pdf_prompt(generated_text)
    
    # Save files
    csv_file_path = save_to_file("data.csv", csv_data)
    py_file_path = save_to_file("pdf_code.py", python_code)
    
    # Execute Python code for PDF generation
    result = execute_python_code(csv_file_path, py_file_path, 'pdf')
    
    if isinstance(result, Exception):
        logger.error(f"PDF generation error: {result}")
        return HttpResponse(str(result), status=500)
    else:
        return HttpResponse(generated_text, content_type='text/plain')

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
    """Process CSV with different output types."""
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        csv_file_path = data.get('file_path', '')
        output_type = data.get('output_type', 'csv')  # csv, image, pdf
        
        script_path = get_script_path(output_type)
        
        try:
            # Execute the Python script based on output type
            results = execute_python_code(csv_file_path, script_path, output_type)
            
            if isinstance(results, Exception):
                return JsonResponse({'error': str(results)}, status=500)
            
            # Generate ID and cache results
            result_id = str(uuid.uuid4())
            
            # Store different types of results
            if output_type == 'csv':
                cache.set(f'result_{result_id}', results, timeout=3600)
                preview_data = results[:1000] + '...' if len(results) > 1000 else results
            elif output_type == 'image':
                cache.set(f'result_{result_id}', results, timeout=3600)  # results is image data
                preview_data = "Image generated successfully"
            elif output_type == 'pdf':
                cache.set(f'result_{result_id}', results, timeout=3600)  # results is PDF data
                preview_data = "PDF report generated successfully"
            
            return JsonResponse({
                'success': True,
                'result_id': result_id,
                'output_type': output_type,
                'preview_data': preview_data
            })
            
        except Exception as e:
            logger.error(f"Error processing {output_type}: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return render(request, 'gpt_interface.html')

def get_script_path(output_type):
    """Get the script path based on output type."""
    if output_type == 'csv':
        return os.path.join(default_storage.location, "code.py")
    elif output_type == 'image':
        return os.path.join(default_storage.location, "image_code.py")
    elif output_type == 'pdf':
        return os.path.join(default_storage.location, "pdf_code.py")
    else:
        return os.path.join(default_storage.location, "code.py")

@csrf_exempt
def preview_result(request):
    """Preview processing results based on output type."""
    if request.method == 'GET':
        result_id = request.GET.get('result_id')
        output_type = request.GET.get('output_type', 'csv')
        
        if not result_id:
            return JsonResponse({'error': 'No result ID provided'}, status=400)
        
        result_data = cache.get(f'result_{result_id}')
        if not result_data:
            return JsonResponse({'error': 'Result not found or expired'}, status=404)
        
        try:
            if output_type == 'csv':
                return preview_csv_result(result_data, result_id)
            elif output_type == 'image':
                return preview_image_result(result_data, result_id)
            elif output_type == 'pdf':
                return preview_pdf_result(result_data, result_id)
            else:
                return JsonResponse({'error': 'Invalid output type'}, status=400)
                
        except Exception as e:
            logger.error(f"Error previewing {output_type} result: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def preview_csv_result(result_data, result_id):
    """Preview CSV result data."""
    try:
        df = pd.read_csv(StringIO(result_data))
        preview_info = {
            'success': True,
            'result_id': result_id,
            'output_type': 'csv',
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'sample_data': df.head(10).fillna('').to_dict('records'),
            'data_types': df.dtypes.astype(str).to_dict(),
            'file_size': len(result_data)
        }
        return JsonResponse(preview_info)
    except Exception as e:
        return JsonResponse({
            'success': True,
            'result_id': result_id,
            'output_type': 'csv',
            'raw_preview': result_data[:2000] + '...' if len(result_data) > 2000 else result_data,
            'file_size': len(result_data)
        })

def preview_image_result(result_data, result_id):
    """Preview image result."""
    # result_data should be base64 encoded image
    preview_info = {
        'success': True,
        'result_id': result_id,
        'output_type': 'image',
        'image_data': result_data,  # base64 encoded image
        'message': 'Image generated successfully'
    }
    return JsonResponse(preview_info)

def preview_pdf_result(result_data, result_id):
    """Preview PDF result."""
    preview_info = {
        'success': True,
        'result_id': result_id,
        'output_type': 'pdf',
        'file_size': len(result_data),
        'message': 'PDF report generated successfully'
    }
    return JsonResponse(preview_info)

@csrf_exempt
def download_result(request):
    """Download processing results based on type."""
    if request.method == 'GET':
        result_id = request.GET.get('result_id')
        output_type = request.GET.get('output_type', 'csv')
        
        if not result_id:
            return JsonResponse({'error': 'No result ID provided'}, status=400)
        
        result_data = cache.get(f'result_{result_id}')
        if not result_data:
            return JsonResponse({'error': 'Result not found or expired'}, status=404)
        
        try:
            if output_type == 'csv':
                response = HttpResponse(result_data, content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="processed_data_{result_id[:8]}.csv"'
            elif output_type == 'image':
                # Decode base64 image data
                image_data = base64.b64decode(result_data)
                response = HttpResponse(image_data, content_type='image/png')
                response['Content-Disposition'] = f'attachment; filename="generated_chart_{result_id[:8]}.png"'
            elif output_type == 'pdf':
                response = HttpResponse(result_data, content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="analysis_report_{result_id[:8]}.pdf"'
            else:
                return JsonResponse({'error': 'Invalid output type'}, status=400)
            
            logger.info(f"Result {result_id} ({output_type}) downloaded successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error downloading {output_type} result: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def process_prompt(generated_text):
    """Process the prompt to generate CSV data and corresponding code snippets."""
    csv_data = ''
    sas_code = ''
    python_code = ''

    sections = generated_text.split('```')
    for i, section in enumerate(sections):
        section = section.strip()
        if i == 1:
            csv_data = section
        elif i == 3 and 'sas' in sections[i-1].lower():
            sas_code_lines = section.split('\n')
            sas_code = '\n'.join(sas_code_lines[1:]).strip()
        elif i == 5 and 'python' in sections[i-1].lower():
            python_code_lines = section.split('\n')
            python_code = '\n'.join(python_code_lines[1:]).strip()

    logger.info(f"CSV data: {csv_data}")
    logger.info(f"SAS code: {sas_code}")
    logger.info(f"Python code: {python_code}")

    return csv_data, sas_code, python_code

def process_image_prompt(generated_text):
    """Process prompt for image generation."""
    csv_data = ''
    python_code = ''

    sections = generated_text.split('```')
    for i, section in enumerate(sections):
        section = section.strip()
        if i == 1:
            csv_data = section
        elif 'python' in sections[i-1].lower() and ('matplotlib' in section or 'plt.' in section or 'seaborn' in section):
            python_code_lines = section.split('\n')
            python_code = '\n'.join(python_code_lines[1:]).strip()

    return csv_data, python_code

def process_pdf_prompt(generated_text):
    """Process prompt for PDF report generation."""
    csv_data = ''
    python_code = ''

    sections = generated_text.split('```')
    for i, section in enumerate(sections):
        section = section.strip()
        if i == 1:
            csv_data = section
        elif 'python' in sections[i-1].lower() and ('reportlab' in section or 'pdf' in section.lower()):
            python_code_lines = section.split('\n')
            python_code = '\n'.join(python_code_lines[1:]).strip()

    return csv_data, python_code

def save_to_file(file_name, content):
    """Save the given content to a file with the specified name."""
    if not content.strip():
        logger.warning(f"No content to save for {file_name}")
        return None

    file_path = os.path.join(default_storage.location, file_name)

    with default_storage.open(file_path, 'w') as f:
        f.write(content)

    if default_storage.exists(file_path):
        logger.info(f"Saved content to {file_path}")
    else:
        logger.error(f"Failed to save content to {file_name}")

    return file_path

def execute_python_code(csv_file_path, py_file_path, output_type='csv'):
    """Execute the Python code on the CSV data with different output types."""
    try:
        if csv_file_path is None or py_file_path is None:
            logger.error("CSV file or Python file path is None.")
            raise ValueError("CSV file or Python file path is None.")

        with default_storage.open(csv_file_path, 'r') as csv_file:
            csv_data = csv_file.read()

        if not csv_data.strip():
            logger.error("CSV file is empty.")
            raise ValueError("CSV file is empty.")

        df = pd.read_csv(StringIO(csv_data))
        df = df.astype(str)

        with default_storage.open(py_file_path, 'r') as py_file:
            python_code = py_file.read()

        logger.info(f"Executing Python code for {output_type}: {python_code}")

        # Set up execution environment based on output type
        exec_globals = setup_execution_environment(df, output_type)

        try:
            exec(python_code, exec_globals)
            return handle_execution_result(exec_globals, output_type)

        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return e

    except Exception as e:
        logger.error(f"Error executing Python code on CSV data: {e}")
        return e

def setup_execution_environment(df, output_type):
    """Set up the execution environment based on output type."""
    base_globals = {
        'df': df,
        'pd': pd,
        'StringIO': StringIO,
        'default_storage': default_storage,
        'processed_data': None
    }
    
    if output_type == 'image':
        base_globals.update({
            'plt': plt,
            'sns': sns,
            'matplotlib': matplotlib,
            'Image': Image,
            'BytesIO': BytesIO,
            'base64': base64,
            'image_data': None
        })
    elif output_type == 'pdf':
        base_globals.update({
            'SimpleDocTemplate': SimpleDocTemplate,
            'Paragraph': Paragraph,
            'Spacer': Spacer,
            'RLImage': RLImage,
            'PageBreak': PageBreak,
            'Table': Table,
            'TableStyle': TableStyle,
            'getSampleStyleSheet': getSampleStyleSheet,
            'ParagraphStyle': ParagraphStyle,
            'letter': letter,
            'A4': A4,
            'inch': inch,
            'colors': colors,
            'tempfile': tempfile,
            'plt': plt,
            'sns': sns,
            'pdf_buffer': BytesIO(),
            'pdf_data': None
        })
    
    return base_globals

def handle_execution_result(exec_globals, output_type):
    """Handle the execution result based on output type."""
    if output_type == 'csv':
        if 'processed_data' in exec_globals and exec_globals['processed_data'] is not None:
            df = exec_globals['processed_data']
        else:
            df = exec_globals['df']
            logger.warning("Processed data not set in executed Python code. Using original DataFrame.")
        
        results_buffer = StringIO()
        df.to_csv(results_buffer, index=False)
        results_buffer.seek(0)
        return results_buffer.getvalue()
    
    elif output_type == 'image':
        if 'image_data' in exec_globals and exec_globals['image_data'] is not None:
            return exec_globals['image_data']  # Should be base64 encoded
        else:
            logger.error("Image data not generated in executed Python code.")
            raise ValueError("Image data not generated in executed Python code.")
    
    elif output_type == 'pdf':
        if 'pdf_data' in exec_globals and exec_globals['pdf_data'] is not None:
            return exec_globals['pdf_data']  # Should be PDF binary data
        else:
            logger.error("PDF data not generated in executed Python code.")
            raise ValueError("PDF data not generated in executed Python code.")
    
    else:
        raise ValueError(f"Unknown output type: {output_type}")
