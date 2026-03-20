# synthetic_data_views.py
# 新增功能：从文本描述生成合成数据

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.core.cache import cache
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# 导入现有的工具函数
from .gpt_backend_utils import call_openai_chat


def synthetic_data_interface(request):
    """渲染合成数据生成界面"""
    # 读取HTML模板
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'synthetic_data_interface.html')
    
    # 如果模板不存在，返回内联HTML
    if not os.path.exists(template_path):
        # 可以直接返回HTML内容
        with open('/home/claude/synthetic_data_interface.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HttpResponse(html_content)
    
    return render(request, 'synthetic_data_interface.html')

@csrf_exempt
def analyze_synthetic_data_request(request):
    """
    分析用户的文本描述，确定应该生成什么样的合成数据
    
    输入：
    - description: 用户的文本描述
    
    输出：
    - 数据分析结果（变量类型、分布、样本量等）
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        description = (data.get('description') or '').strip()
        
        if not description:
            return JsonResponse({'error': 'No description provided'}, status=400)
        
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            return JsonResponse({'error': 'OpenAI API key not configured'}, status=500)
        
        # 调用LLM分析描述
        analysis_result = analyze_data_description(description, api_key)
        
        # 生成唯一ID用于缓存
        analysis_id = str(uuid.uuid4())
        cache.set(f'analysis_{analysis_id}', analysis_result, timeout=3600)
        
        return JsonResponse({
            'success': True,
            'analysis_id': analysis_id,
            'analysis': analysis_result
        }, status=200)
        
    except Exception as e:
        logger.exception("analyze_synthetic_data_request failed")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def generate_synthetic_data(request):
    """
    基于分析结果生成实际的合成数据
    
    输入：
    - analysis_id: 之前分析的ID，或
    - analysis: 直接提供的分析结果
    
    输出：
    - CSV数据（预览和下载）
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        
        # 获取分析结果
        analysis_id = data.get('analysis_id')
        analysis = data.get('analysis')
        
        if analysis_id:
            analysis = cache.get(f'analysis_{analysis_id}')
            if not analysis:
                return JsonResponse({'error': 'Analysis not found or expired'}, status=404)
        elif not analysis:
            return JsonResponse({'error': 'No analysis provided'}, status=400)
        
        # 生成合成数据
        df = create_synthetic_dataframe(analysis)
        
        # 转换为CSV
        csv_string = df.to_csv(index=False)
        
        # 生成预览
        preview_html = generate_data_preview(df, analysis)
        
        # 缓存结果
        result_id = str(uuid.uuid4())
        cache.set(f'synthetic_data_{result_id}', csv_string, timeout=3600)
        
        return JsonResponse({
            'success': True,
            'result_id': result_id,
            'preview_html': preview_html,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns)
        }, status=200)
        
    except Exception as e:
        logger.exception("generate_synthetic_data failed")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def download_synthetic_data(request):
    """下载生成的合成数据"""
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    result_id = request.GET.get('result_id')
    if not result_id:
        return JsonResponse({'error': 'No result ID provided'}, status=400)
    
    csv_data = cache.get(f'synthetic_data_{result_id}')
    if not csv_data:
        return JsonResponse({'error': 'Data not found or expired'}, status=404)
    
    from django.http import HttpResponse
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="synthetic_data_{result_id[:8]}.csv"'
    return response


def analyze_data_description(description: str, api_key: str) -> dict:
    """
    使用LLM分析用户描述，提取数据生成规格
    """
    system_prompt = """You are a data science expert specializing in synthetic data generation.
Analyze the user's description and extract specifications for generating synthetic data.

Return a JSON object with the following structure:
{
    "dataset_name": "descriptive name",
    "sample_size": 100,
    "variables": [
        {
            "name": "variable_name",
            "type": "numeric|categorical|datetime|text",
            "distribution": "normal|uniform|poisson|binomial|categorical|...",
            "parameters": {
                "mean": 50, "std": 10  // for normal
                // OR "min": 0, "max": 100  // for uniform
                // OR "categories": ["A", "B", "C"], "probabilities": [0.5, 0.3, 0.2]
            },
            "description": "what this variable represents"
        }
    ],
    "relationships": [
        {
            "description": "variable X correlates with Y",
            "type": "correlation|causation|...",
            "variables": ["var1", "var2"],
            "strength": 0.7
        }
    ]
}

Be specific and practical. Infer reasonable defaults if not specified."""

    user_prompt = f"""Analyze this data description and provide the JSON specification:

Description: {description}

Extract:
1. How many rows/samples should be generated?
2. What variables/columns are needed?
3. What is the data type and distribution for each variable?
4. Are there any relationships between variables?
5. What are reasonable parameter values?"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = call_openai_chat(
            messages=messages,
            api_key=api_key,
            model='gpt-4o',
            temperature=0.3,
            
        )
        
        # 提取JSON
        #response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(response)
        response_text =response #.choices[0].message.content
        
        # 尝试解析JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            # 如果找不到JSON，返回默认结构
            analysis = {
                "dataset_name": "Generated Dataset",
                "sample_size": 100,
                "variables": [],
                "relationships": [],
                "raw_response": response_text
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing description: {e}")
        raise


def create_synthetic_dataframe(analysis: dict) -> pd.DataFrame:
    """
    根据分析结果生成实际的合成数据DataFrame
    """
    sample_size = analysis.get('sample_size', 100)
    variables = analysis.get('variables', [])
    
    if not variables:
        raise ValueError("No variables specified in analysis")
    
    data = {}
    
    for var in variables:
        var_name = var['name']
        var_type = var['type']
        distribution = var.get('distribution', 'uniform')
        params = var.get('parameters', {})
        
        # 根据类型和分布生成数据
        if var_type == 'numeric':
            data[var_name] = generate_numeric_variable(
                sample_size, distribution, params
            )
        elif var_type == 'categorical':
            data[var_name] = generate_categorical_variable(
                sample_size, distribution, params
            )
        elif var_type == 'datetime':
            data[var_name] = generate_datetime_variable(
                sample_size, distribution, params
            )
        elif var_type == 'text':
            data[var_name] = generate_text_variable(
                sample_size, distribution, params
            )
        else:
            # 默认生成uniform数值
            data[var_name] = np.random.uniform(0, 100, sample_size)
    
    df = pd.DataFrame(data)
    
    # 应用变量间关系
    relationships = analysis.get('relationships', [])
    df = apply_relationships(df, relationships)
    
    return df


def generate_numeric_variable(n: int, distribution: str, params: dict) -> np.ndarray:
    """生成数值型变量"""
    if distribution == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        return np.random.normal(mean, std, n)
    
    elif distribution == 'uniform':
        min_val = params.get('min', 0)
        max_val = params.get('max', 100)
        return np.random.uniform(min_val, max_val, n)
    
    elif distribution == 'poisson':
        lambda_val = params.get('lambda', 5)
        return np.random.poisson(lambda_val, n)
    
    elif distribution == 'exponential':
        scale = params.get('scale', 1.0)
        return np.random.exponential(scale, n)
    
    elif distribution == 'binomial':
        n_trials = params.get('n', 10)
        p = params.get('p', 0.5)
        return np.random.binomial(n_trials, p, n)
    
    else:
        # 默认使用正态分布
        return np.random.normal(50, 15, n)


def generate_categorical_variable(n: int, distribution: str, params: dict) -> np.ndarray:
    """生成分类变量"""
    categories = params.get('categories', ['A', 'B', 'C'])
    probabilities = params.get('probabilities', None)
    
    if probabilities:
        # 归一化概率
        probs = np.array(probabilities)
        probs = probs / probs.sum()
    else:
        probs = None
    
    return np.random.choice(categories, size=n, p=probs)


def generate_datetime_variable(n: int, distribution: str, params: dict) -> pd.Series:
    """生成日期时间变量"""
    start_date = params.get('start_date', '2020-01-01')
    end_date = params.get('end_date', '2023-12-31')
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 生成随机日期
    time_delta = (end - start).days
    random_days = np.random.randint(0, time_delta, n)
    dates = [start + pd.Timedelta(days=int(d)) for d in random_days]
    
    return pd.Series(dates)


def generate_text_variable(n: int, distribution: str, params: dict) -> np.ndarray:
    """生成文本变量"""
    templates = params.get('templates', ['Text_{i}'])
    
    if '{i}' in str(templates[0]):
        # 使用索引模板
        return np.array([templates[0].replace('{i}', str(i)) for i in range(n)])
    else:
        # 从模板中随机选择
        return np.random.choice(templates, size=n)


def apply_relationships(df: pd.DataFrame, relationships: list) -> pd.DataFrame:
    """应用变量间的关系（如相关性）"""
    for rel in relationships:
        rel_type = rel.get('type', '')
        variables = rel.get('variables', [])
        
        if rel_type == 'correlation' and len(variables) == 2:
            strength = rel.get('strength', 0.5)
            var1, var2 = variables
            
            if var1 in df.columns and var2 in df.columns:
                # 简单实现：在var2中添加与var1相关的噪声
                if pd.api.types.is_numeric_dtype(df[var1]) and pd.api.types.is_numeric_dtype(df[var2]):
                    noise = np.random.normal(0, df[var2].std() * (1 - abs(strength)), len(df))
                    df[var2] = strength * df[var1] + (1 - strength) * df[var2] + noise
    
    return df


def generate_data_preview(df: pd.DataFrame, analysis: dict) -> str:
    """生成数据预览HTML"""
    preview_rows = min(10, len(df))
    
    # 基本统计
    stats_html = "<div class='stats-section'><h3>Dataset Overview</h3>"
    stats_html += f"<p><strong>Name:</strong> {analysis.get('dataset_name', 'Synthetic Dataset')}</p>"
    stats_html += f"<p><strong>Rows:</strong> {len(df)} | <strong>Columns:</strong> {len(df.columns)}</p>"
    stats_html += "</div>"
    
    # 变量描述
    vars_html = "<div class='variables-section'><h3>Variables</h3><ul>"
    for var in analysis.get('variables', []):
        vars_html += f"<li><strong>{var['name']}</strong> ({var['type']}): {var.get('description', 'N/A')}"
        vars_html += f"<br><small>Distribution: {var.get('distribution', 'N/A')}</small></li>"
    vars_html += "</ul></div>"
    
    # 数据预览表格
    table_html = "<div class='preview-section'><h3>Data Preview (first 10 rows)</h3>"
    table_html += df.head(preview_rows).to_html(classes='table table-striped', index=False)
    table_html += "</div>"
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_html = "<div class='statistics-section'><h3>Descriptive Statistics</h3>"
        desc_html += df[numeric_cols].describe().to_html(classes='table table-bordered')
        desc_html += "</div>"
    else:
        desc_html = ""
    
    full_html = f"""
    <div class='synthetic-data-preview'>
        <style>
            .synthetic-data-preview {{ font-family: Arial, sans-serif; }}
            .stats-section, .variables-section, .preview-section, .statistics-section {{
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
            .table th {{ background-color: #f5f5f5; font-weight: bold; }}
            .table-striped tbody tr:nth-child(odd) {{ background-color: #f9f9f9; }}
            h3 {{ color: #333; margin-top: 0; }}
            ul {{ padding-left: 20px; }}
            li {{ margin: 8px 0; }}
        </style>
        {stats_html}
        {vars_html}
        {table_html}
        {desc_html}
    </div>
    """
    
    return full_html
