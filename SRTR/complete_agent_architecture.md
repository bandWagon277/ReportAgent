# 完整 Agent 架构与用户输入信息流详解

## 一、完整 Agent 分类体系

### Agent 0 - 路由与匹配Agent (Router & Matcher Agent)

**核心职责**: 意图识别、变量名提取、字典文件匹配

**功能模块**:

1. **变量名提取模块**
   - `extract_variables_from_query(query)` - 使用正则提取大写变量名
   - 模式: `\b[A-Z][A-Z0-9_]{2,}\b`
   - 优先级: 带下划线 > 纯大写字母

2. **字典索引管理**
   - `build_dictionaries_index()` - 扫描所有CSV字典文件
   - `load_dictionaries_index()` - 加载或重建索引
   - 索引结构:
     ```json
     {
       "files": [...],
       "variables": {
         "CAN_ABO": ["/path/to/dict1.csv", "/path/to/dict2.csv"],
         "DON_AGE": ["/path/to/dict3.csv"]
       },
       "total_variables": 5000
     }
     ```

3. **智能路径匹配**
   - `guess_paths_for_variable(var_name, index, topk=3)` - 多策略匹配
   - 匹配策略(优先级递减):
     1. **精确匹配**: 变量名在索引中直接命中
     2. **部分匹配**: 变量名前缀匹配
     3. **文件名匹配**: CSV文件名包含变量名

4. **字典内容读取**
   - `read_dictionary_with_fallback(csv_path)` - 多编码读取CSV
   - 支持编码: utf-8-sig, utf-8, latin-1, cp1252
   - 返回格式化表格字符串

5. **意图分类**
   - `classify_intent_by_pattern(query)` - 基于关键词的意图识别
   - 支持意图类型:
     - `definition`: "what is", "define", "meaning"
     - `calculation`: "calculate", "estimate", "predict"
     - `analysis`: "analyze", "compare", "trend"
     - `visualization`: "chart", "plot", "graph"
     - `data_lookup`: "find", "search", "show me"

---

### Agent A - 文档检索与问答Agent (Retrieval & QA Agent)

**核心职责**: RAG检索、文档问答、概念解释

**功能模块**:

1. **文档索引构建**
   - `build_documents_index()` - 扫描DOCS_DIR下所有HTML文件
   - `parse_srtr_html(html_path)` - 解析SRTR官网HTML镜像
   - `chunk_document(doc, chunk_size=500, overlap=100)` - 文档分块

2. **向量化与存储**
   - `_get_embedding(text)` - 调用OpenAI Embedding API
   - 模型: `text-embedding-3-small`
   - 存储格式: JSON文件(每个chunk一个.json)

3. **语义检索**
   - `retrieve_chunks(query, filters=None, top_k=5)` - 向量相似度检索
   - 计算公式: 余弦相似度
   - 过滤器: section_type, document_id

4. **上下文构建与问答**
   - `answer_question(query)` - 主流程函数
   - Prompt模板:
     ```
     You are a helpful assistant for SRTR data questions.
     
     Context from official documentation:
     {retrieved_chunks}
     
     User question: {query}
     
     Provide accurate answer based on context.
     ```
   - 温度: 0.3(偏向事实性)



### Agent B - 计算器Agent (Calculator Agent)

**核心职责**: 医学公式计算、参数提取

**功能模块**:

1. **参数提取**
   - `_extract_calculator_parameters(query, tool)` - 从自然语言提取参数
   - Prompt示例:
     ```
     Extract these parameters from query:
     - age (integer)
     - blood_type (A/B/AB/O)
     - dialysis_time (months)
     
     Query: "Patient is 45 years old, blood type B, on dialysis for 24 months"
     
     Return JSON format.
     ```

2. **计算工具**
   - `calculate_kidney_waiting_time(params)` - 肾移植等待时间预测
   - 支持参数:
     - age, blood_type, dialysis_time
     - pra_level, hla_mismatch, diabetes
     - previous_transplant

3. **结果验证**
   - 参数范围检查
   - 必填字段验证
   - 单位转换

---

### Agent C - 文本分析Agent (Text Analysis Agent)

**核心职责**: 处理非结构化文本、代码分析

**功能模块**:

1. **文件类型检测**
   - `detect_file_type(filename)` - 基于扩展名
   - 支持类型: csv, text, image, pdf, document

2. **多编码读取**
   - `read_text_file(file_path, max_chars=50000)` - 智能编码检测
   - 截断保护: 超过50000字符自动截断

3. **GPT分析**
   - `analyze_text_with_gpt(text_content, user_prompt, api_key, filename)`
   - 不执行代码,仅提供分析和建议
   - 温度: 0.7(允许创造性分析)

4. **代码提取**
   - `extract_code_for_execution(response_text, target_language='python')`
   - 支持多语言代码块识别

---

## 二、主入口与路由逻辑

### 核心端点: `api_query(request)`

**位置**: views.py (未在提供的代码中完整展示,但从上下文推断)

**路由决策树**:

```python
def api_query(request):
    """
    主查询入口 - 根据用户输入路由到不同Agent
    """
    # 1. 解析请求
    data = json.loads(request.body)
    query = data.get("query", "").strip()
    
    # 2. 提取变量名 (Agent 0)
    variables = extract_variables_from_query(query)
    
    # 3. 意图分类 (Agent 0)
    intent = classify_intent_by_pattern(query)
    
    # 4. 路由决策
    if variables and "definition" in intent:
        # 有变量名 + 定义类问题 → 字典查询
        return route_to_dictionary_lookup(query, variables)
    
    elif "calculation" in intent or "estimate" in query.lower():
        # 计算类问题 → Calculator Agent
        return route_to_calculator(query)
    
    elif "document" in data or data.get("context_mode") == "rag":
        # 文档相关 → RAG Agent
        return route_to_rag_agent(query)
    
    elif data.get("file_type") in ["text", "code"]:
        # 文本分析 → Text Analysis Agent
        return route_to_text_agent(query, data.get("file_content"))
    
    else:
        # 通用问答 → RAG Agent
        return route_to_rag_agent(query)
```

---

## 三、用户输入完整信息流

### 场景1: 变量定义查询

**用户输入**:
```
"What does CAN_ABO mean in the SRTR database?"
```

**信息流**:

```
[1] 请求接收
    ↓
    api_query(request)  # views.py
    ├─ 解析JSON: query = "What does CAN_ABO mean..."
    └─ 提取原始query字符串

[2] Agent 0: 路由与匹配
    ↓
    extract_variables_from_query(query)  # views.py ~line 300
    ├─ 正则匹配: re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", query)
    ├─ 发现: ["CAN_ABO"]
    └─ 返回: variables = ["CAN_ABO"]
    
    ↓
    classify_intent_by_pattern(query)  # views.py ~line 400
    ├─ 检测关键词: "what", "mean" → intent = "definition"
    └─ 返回: intent = "definition"
    
    ↓
    决策: variables存在 + intent="definition" → 路由到字典查询

[3] 字典匹配与读取
    ↓
    load_dictionaries_index()  # views.py line 203
    ├─ 检查索引文件: DATA_REPO/meta/dictionaries.index.json
    ├─ 如果不存在 → build_dictionaries_index()
    │   ├─ 扫描: DATA_REPO/dictionaries/**/*.csv
    │   ├─ 读取每个CSV的"Variable"列
    │   ├─ 构建映射: {"CAN_ABO": ["/path/to/General.csv"], ...}
    │   └─ 保存索引
    └─ 返回: index = {files: [...], variables: {...}}
    
    ↓
    guess_paths_for_variable("CAN_ABO", index, topk=3)  # views.py line 220
    ├─ 精确匹配: "CAN_ABO" in index["variables"]
    ├─ 命中: DATA_REPO/dictionaries/General/Candidate.csv
    ├─ 评分: score=1.0, reason="exact_match"
    └─ 返回: [("/.../Candidate.csv", 1.0, "exact_match")]
    
    ↓
    read_dictionary_with_fallback(csv_path)  # views.py ~line 260
    ├─ 尝试编码: utf-8-sig → 成功
    ├─ 解析CSV: DictReader(content)
    ├─ 查找行: Variable == "CAN_ABO"
    ├─ 提取: {
    │     "Variable": "CAN_ABO",
    │     "Type": "Char",
    │     "Length": "2",
    │     "Format": "$2.",
    │     "Label": "ABO blood type"
    │   }
    └─ 格式化为表格字符串

[4] 响应构建
    ↓
    return JsonResponse({
        "query": "What does CAN_ABO mean...",
        "intent": "definition",
        "variable": "CAN_ABO",
        "matched_file": "Candidate.csv",
        "definition": "Variable: CAN_ABO\nType: Char\nLabel: ABO blood type",
        "confidence": 1.0
    })

[5] 前端渲染
    ├─ 显示变量名: CAN_ABO
    ├─ 显示定义表格
    └─ 显示来源文件
```

---

### 场景2: 计算器调用

**用户输入**:
```
"Calculate kidney waiting time for a 45-year-old patient with blood type B who has been on dialysis for 24 months"
```

**信息流**:

```
[1] 请求接收
    ↓
    api_query(request) 或 api_calculate(request)  # views.py line 2096

[2] 意图识别
    ↓
    classify_intent_by_pattern(query)
    ├─ 检测: "calculate" → intent = "calculation"
    └─ 决策: 路由到Calculator Agent

[3] 参数提取
    ↓
    _extract_calculator_parameters(query, tool="kidney_waiting_time")
    # views.py ~line 1800 (推测位置)
    
    ├─ 构建Prompt:
    │   """
    │   Extract these parameters:
    │   - age (integer)
    │   - blood_type (A/B/AB/O)
    │   - dialysis_time (months)
    │   
    │   From: "Calculate kidney waiting time for a 45-year-old..."
    │   
    │   Return JSON: {"age": ..., "blood_type": ..., ...}
    │   """
    │
    ├─ 调用GPT-4o:
    │   _openai_chat(messages, temperature=0.0)  # views.py line 29
    │   ├─ URL: https://api.openai.com/v1/chat/completions
    │   ├─ Model: gpt-4o
    │   └─ Temperature: 0.0 (精确提取)
    │
    └─ 解析响应:
        {
          "age": 45,
          "blood_type": "B",
          "dialysis_time": 24,
          "extracted_info": "Successfully extracted 3/3 parameters",
          "missing_parameters": []
        }

[4] 计算执行
    ↓
    calculate_kidney_waiting_time(params)  # views.py ~line 1900
    ├─ 验证参数:
    │   ├─ age: 18-100 ✓
    │   ├─ blood_type: A/B/AB/O ✓
    │   └─ dialysis_time: ≥0 ✓
    │
    ├─ 应用公式:
    │   base_time = 365 * 3  # 基准3年
    │   age_factor = (age - 40) * 0.1 if age > 40 else 0
    │   blood_factor = {"O": 1.5, "B": 1.2, "A": 1.0, "AB": 0.8}[blood_type]
    │   dialysis_factor = min(dialysis_time / 12 * 0.1, 0.5)
    │   
    │   estimated_days = base_time * (1 + age_factor) * blood_factor * (1 - dialysis_factor)
    │
    └─ 返回结果:
        {
          "estimated_waiting_days": 1460,
          "estimated_waiting_years": 4.0,
          "factors": {
            "age_impact": "+5%",
            "blood_type_impact": "+20%",
            "dialysis_impact": "-20%"
          },
          "confidence": "moderate",
          "disclaimer": "This is an estimate based on historical data..."
        }

[5] 响应返回
    ↓
    return JsonResponse({
        "tool": "kidney_waiting_time",
        "input_parameters": {"age": 45, "blood_type": "B", ...},
        "result": {...},
        "status": "success"
    })

[6] 前端显示
    ├─ 显示等待时间: 4.0 years (1460 days)
    ├─ 显示影响因素分解
    └─ 显示免责声明
```

---

### 场景3: RAG文档问答

**用户输入**:
```
"How is the KDPI score calculated?"
```

**信息流**:

```
[1] 请求接收与路由
    ↓
    api_query(request)
    ├─ 无明确变量名
    ├─ intent: "question" (默认)
    └─ 决策: 路由到RAG Agent

[2] 文档索引加载
    ↓
    load_chunks_index()  # views.py ~line 800
    ├─ 读取: DATA_REPO/meta/chunks.index.json
    ├─ 包含所有chunk的元数据:
    │   [
    │     {
    │       "chunk_id": "doc1_chunk3",
    │       "text": "KDPI calculation involves 10 factors...",
    │       "section_title": "KDPI Score",
    │       "document_id": "kdri_kdpi_guide",
    │       "embedding_path": "/.../embeddings/doc1_chunk3.json"
    │     },
    │     ...
    │   ]
    └─ 返回: chunks_index = [...]

[3] 查询向量化
    ↓
    _get_embedding(query)  # views.py line 53
    ├─ 调用OpenAI API:
    │   POST https://api.openai.com/v1/embeddings
    │   {
    │     "model": "text-embedding-3-small",
    │     "input": "How is the KDPI score calculated?"
    │   }
    │
    └─ 返回: query_embedding = [0.012, -0.034, 0.056, ..., 0.021]  # 1536维

[4] 向量检索
    ↓
    retrieve_chunks(query, filters=None, top_k=5)  # views.py ~line 900
    ├─ 遍历所有chunks:
    │   for chunk in chunks_index:
    │       # 读取chunk的embedding
    │       chunk_emb = load_embedding(chunk["embedding_path"])
    │       
    │       # 计算余弦相似度
    │       similarity = cosine_similarity(query_embedding, chunk_emb)
    │       # = dot(q, c) / (norm(q) * norm(c))
    │       
    │       candidates.append((chunk, similarity))
    │
    ├─ 排序: sorted(candidates, key=lambda x: x[1], reverse=True)
    │
    └─ 返回Top-5:
        [
          {
            "chunk_id": "doc1_chunk3",
            "text": "KDPI calculation involves 10 donor factors...",
            "section_title": "KDPI Score",
            "similarity": 0.87
          },
          {
            "chunk_id": "doc2_chunk15",
            "text": "The formula for KDPI includes age, height...",
            "section_title": "Donor Risk Factors",
            "similarity": 0.82
          },
          ...
        ]

[5] 上下文构建与问答
    ↓
    answer_question(query)  # views.py ~line 1100
    ├─ 构建Prompt:
    │   system_msg = """
    │   You are a helpful assistant for SRTR transplant data questions.
    │   Answer based on the provided context from official documentation.
    │   If context doesn't contain the answer, say so clearly.
    │   """
    │   
    │   context = "\n\n".join([
    │       f"[Source: {c['section_title']}]\n{c['text']}"
    │       for c in retrieved_chunks
    │   ])
    │   
    │   user_msg = f"""
    │   Context:
    │   {context}
    │   
    │   Question: {query}
    │   
    │   Please provide a clear answer based on the context.
    │   """
    │
    ├─ 调用GPT-4o:
    │   _openai_chat(messages=[
    │       {"role": "system", "content": system_msg},
    │       {"role": "user", "content": user_msg}
    │   ], temperature=0.3)
    │
    └─ GPT响应:
        """
        The KDPI (Kidney Donor Profile Index) is calculated using 10 donor factors:
        
        1. Age
        2. Height
        3. Weight
        4. Ethnicity
        5. History of hypertension
        6. History of diabetes
        7. Cause of death
        8. Serum creatinine
        9. Hepatitis C status
        10. Donation after circulatory death (DCD) status
        
        The formula generates a value from 0-100%, where lower scores indicate 
        better quality kidneys with longer expected graft survival.
        
        [Sources: KDPI Score, Donor Risk Factors]
        """

[6] 响应返回
    ↓
    return JsonResponse({
        "query": "How is the KDPI score calculated?",
        "answer": "The KDPI (Kidney Donor Profile Index) is calculated...",
        "sources": [
            {"section": "KDPI Score", "similarity": 0.87},
            {"section": "Donor Risk Factors", "similarity": 0.82}
        ],
        "retrieval_method": "vector_search",
        "chunks_used": 5
    })

[7] 前端渲染
    ├─ 显示答案(支持Markdown)
    ├─ 显示来源引用(可点击)
    └─ 显示相关性分数
```

---


## 四、函数调用链总结

### 变量查询流程函数链

```
用户输入
  ↓
api_query(request)
  ↓
extract_variables_from_query(query)
  ↓
classify_intent_by_pattern(query)
  ↓
load_dictionaries_index()
  ├─ build_dictionaries_index() (如果索引不存在)
  │   └─ 扫描 DICT_ROOT/**/*.csv
  └─ 返回 index
  ↓
guess_paths_for_variable(var_name, index, topk=3)
  ├─ 精确匹配检查
  ├─ 部分匹配检查
  └─ 文件名匹配
  ↓
read_dictionary_with_fallback(csv_path)
  ├─ 尝试多种编码读取
  └─ 解析CSV并格式化
  ↓
JsonResponse(结果)
```

### 计算器调用函数链

```
用户输入
  ↓
api_query(request) 或 api_calculate(request)
  ↓
classify_intent_by_pattern(query)
  ↓
_extract_calculator_parameters(query, tool)
  ├─ 构建参数提取Prompt
  └─ _openai_chat(messages, temperature=0.0)
  ↓
calculate_kidney_waiting_time(params)
  ├─ 验证参数范围
  ├─ 应用计算公式
  └─ 返回结果dict
  ↓
JsonResponse(结果)
```

### RAG问答函数链

```
用户输入
  ↓
api_query(request)
  ↓
load_chunks_index()
  ├─ 读取 CHUNKS_INDEX_PATH
  └─ 返回所有chunk元数据
  ↓
_get_embedding(query)
  ├─ POST /v1/embeddings
  └─ 返回query向量
  ↓
retrieve_chunks(query, filters=None, top_k=5)
  ├─ 遍历所有chunks
  ├─ 加载每个chunk的embedding
  ├─ 计算余弦相似度
  ├─ 排序取Top-K
  └─ 返回相关chunks
  ↓
answer_question(query)
  ├─ 构建上下文Prompt
  ├─ _openai_chat(messages, temperature=0.3)
  └─ 返回答案
  ↓
JsonResponse(结果)
```



## 五、关键配置路径

```python
# 数据仓库根目录
DATA_REPO = Path(r"C:/Users/18120/Desktop/OPENAIproj/SRTR/data_repo")

# 子目录结构
HTML_MIRROR_DIR = DATA_REPO / "html_mirrors"     # HTML镜像文件
CHUNKS_DIR = DATA_REPO / "chunks"                # 文档分块
EMBEDDINGS_DIR = DATA_REPO / "embeddings"        # 向量存储
DICT_ROOT = DATA_REPO / "dictionaries"           # 数据字典CSV
  ├─ General/
  │   ├─ Candidate.csv      # 候选者变量
  │   ├─ Donor.csv          # 捐赠者变量
  │   └─ Transplant.csv     # 移植变量
  ├─ Kidney_Pancreas/
  └─ Heart_Lung/
CONCEPTS_DIR = DATA_REPO / "concepts"            # R/Python分析项目
DOCS_DIR = DATA_REPO / "docs"                    # 官方文档

# 索引文件
DOCS_INDEX_PATH = DATA_REPO / "meta" / "documents.index.json"
CHUNKS_INDEX_PATH = DATA_REPO / "meta" / "chunks.index.json"
DICT_INDEX = DATA_REPO / "meta" / "dictionaries.index.json"

# Prompt模板
PROMPT_BASE_PATH = r"C:/Users/18120/Desktop/OPENAIproj/"
  ├─ Instruction_prompt_csv.txt
  ├─ Instruction_prompt_image.txt
  ├─ Instruction_prompt_pdf_A.txt
  └─ Instruction_prompt_pdf_B.txt
```

---

## 六、总结

该系统通过**6个专门化Agent**协同工作:

| Agent | 职责 | 核心函数 | 输入 | 输出 |
|-------|------|---------|------|------|
| **Agent 0** | 路由匹配 | `extract_variables_from_query`<br>`guess_paths_for_variable` | 用户query | 变量名、文件路径 |
| **Agent A** | 文档RAG | `retrieve_chunks`<br>`answer_question` | query | 带引用的答案 |
| **Agent B** | 数据分析 | `execute_python_code`<br>`handle_execution_result` | query + CSV | CSV/base64图片 |
| **Agent C** | PDF报告 | `get_pdf_dual_prompts`<br>`execute_python_code` | query + CSV | PDF二进制 |
| **Agent D** | 计算器 | `_extract_calculator_parameters`<br>`calculate_kidney_waiting_time` | query | 计算结果JSON |
| **Agent E** | 文本分析 | `read_text_file`<br>`analyze_text_with_gpt` | 文本文件 | 分析结果 |

**信息流核心路径**:
```
用户输入 
  → 意图识别(Agent 0) 
  → 路由决策 
  → 专门Agent处理 
  → GPT-4o生成/检索 
  → 结果验证/格式化 
  → JSON响应 
  → 前端渲染
```

所有Agent共享:
- OpenAI API封装(`_openai_chat`, `_get_embedding`)
- 代码清理工具(`sanitize_python`)
- 文件存储抽象(`default_storage`)
- 统一错误处理机制
