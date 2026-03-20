
# ==========
# HTML Parser Layer
# ==========
from bs4 import BeautifulSoup
import re
from data_manager import load_documents_index
from utils.constants import HTML_MIRROR_DIR
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
def parse_metric_definition_html(html: str) -> List[Dict[str, Any]]:
    """
    Parse metrics definition HTML page.
    Returns list of metric definitions with structure:
    [
      {
        "metric_code": "1YR_PAT_SURV",
        "title": "1-year patient survival",
        "definition": "Definition text...",
        "notes": "Additional notes..."
      },
      ...
    ]
    """
    soup = BeautifulSoup(html, 'html.parser')
    metrics = []
    
    # Example parsing logic - adjust based on actual HTML structure
    metric_sections = soup.find_all('div', class_='metric-definition')
    
    for section in metric_sections:
        try:
            metric = {
                "metric_code": section.get('data-metric-code', ''),
                "title": section.find('h3').text.strip() if section.find('h3') else '',
                "definition": section.find('div', class_='definition').text.strip() if section.find('div', class_='definition') else '',
                "notes": section.find('div', class_='notes').text.strip() if section.find('div', class_='notes') else ''
            }
            metrics.append(metric)
        except Exception as e:
            logger.warning(f"Failed to parse metric section: {e}")
            continue
    
    return metrics

def parse_wait_time_html(html: str) -> Dict[str, Any]:
    """
    Parse wait time calculator/methodology page.
    Returns structured data about the wait time model.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    result = {
        "title": "",
        "overview": "",
        "input_variables": [],
        "methodology": "",
        "interpretation": ""
    }
    
    # Adjust based on actual HTML structure
    if soup.find('h1'):
        result["title"] = soup.find('h1').text.strip()
    
    overview_section = soup.find('div', class_='overview')
    if overview_section:
        result["overview"] = overview_section.text.strip()
    
    # Parse input variables table
    variables_table = soup.find('table', class_='input-variables')
    if variables_table:
        for row in variables_table.find_all('tr')[1:]:  # Skip header
            cols = row.find_all('td')
            if len(cols) >= 2:
                result["input_variables"].append({
                    "name": cols[0].text.strip(),
                    "description": cols[1].text.strip()
                })
    
    return result

def parse_center_html(html: str) -> Dict[str, Any]:
    """
    Parse transplant center page.
    Returns center information and key metrics.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    result = {
        "center_name": "",
        "center_id": "",
        "location": "",
        "organ_programs": [],
        "metrics": []
    }
    
    # Adjust based on actual HTML structure
    if soup.find('h1', class_='center-name'):
        result["center_name"] = soup.find('h1', class_='center-name').text.strip()
    
    # Parse metrics table
    metrics_table = soup.find('table', class_='center-metrics')
    if metrics_table:
        for row in metrics_table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                result["metrics"].append({
                    "metric": cols[0].text.strip(),
                    "value": cols[1].text.strip(),
                    "national_avg": cols[2].text.strip()
                })
    
    return result

def parse_generic_html(html: str) -> Dict[str, Any]:
    """
    Generic HTML parser for pages without specific structure.
    Extracts title, main content sections, and links.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer"]):
        script.decompose()
    
    result = {
        "title": soup.find('title').text.strip() if soup.find('title') else "",
        "h1": soup.find('h1').text.strip() if soup.find('h1') else "",
        "sections": []
    }
    
    # Extract main content sections
    main_content = soup.find('main') or soup.find('div', class_='content') or soup.body
    if main_content:
        for section in main_content.find_all(['section', 'div'], class_=re.compile(r'section|content-block')):
            section_title = section.find(['h2', 'h3'])
            section_text = section.get_text(strip=True, separator=' ')
            
            if section_text:
                result["sections"].append({
                    "title": section_title.text.strip() if section_title else "",
                    "content": section_text[:1000]  # Limit length
                })
    
    return result

def load_and_parse_html(doc_id: str, doc_type: str) -> Dict[str, Any]:
    """
    Load HTML file and parse based on document type.
    
    Args:
        doc_id: Document identifier
        doc_type: Type of document (metric_definition, wait_time, center_page, etc.)
    
    Returns:
        Parsed structured data
    """
    # Load document metadata
    docs_index = load_documents_index()
    if doc_id not in docs_index:
        raise ValueError(f"Document {doc_id} not found in index")
    
    doc_meta = docs_index[doc_id]
    html_path = HTML_MIRROR_DIR / doc_meta["local_path"]
    
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse based on document type
    if doc_type == "metric_definition":
        return parse_metric_definition_html(html_content)
    elif doc_type == "wait_time":
        return parse_wait_time_html(html_content)
    elif doc_type == "center_page":
        return parse_center_html(html_content)
    else:
        return parse_generic_html(html_content)
