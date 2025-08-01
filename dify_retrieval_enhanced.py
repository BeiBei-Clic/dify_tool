import requests
import json
from typing import List, Dict, Any, Optional

# Dify 配置常量
DIFY_BASE_URL = 'http://localhost'
ABSTRACT_DATASET_ID = '5a36aa21-0aa7-433e-bdab-72aa9931b543'
FULL_TEXT_DATASET_ID = '8e162a14-c930-40a9-8395-39502b58a45b'
DIFY_API_KEY = 'dataset-nn9K2CMUXa9rSKLlNpMwmHU7'

def retrieve_from_abstract_dataset(query: str, top_k: int = 5) -> Dict[str, Any]:
    """从abstract dataset中检索内容"""
    url = f'{DIFY_BASE_URL}/v1/datasets/{ABSTRACT_DATASET_ID}/retrieve'
    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "conversation_id": "",
        "user": "1",
        "retrieval_setting": {
            "top_k": top_k
        }
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    if response.status_code == 200:
        return response.json()
    return {}

def get_document_by_case_id(case_id: str) -> Optional[str]:
    """根据case_id从full text dataset中获取完整文档内容"""
    # 首先获取所有文档列表
    url = f'{DIFY_BASE_URL}/v1/datasets/{FULL_TEXT_DATASET_ID}/documents'
    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json',
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        return None
    
    documents_data = response.json()
    documents = documents_data.get('data', [])
    
    # 查找匹配case_id的文档
    target_document_id = None
    for doc in documents:
        doc_metadata = doc.get('doc_metadata', [])
        for metadata in doc_metadata:
            if metadata.get('name') == 'case_id' and metadata.get('value') == str(case_id):
                target_document_id = doc.get('id')
                break
        if target_document_id:
            break
    
    if not target_document_id:
        return None
    
    # 获取文档的所有segments
    segments_url = f'{DIFY_BASE_URL}/v1/datasets/{FULL_TEXT_DATASET_ID}/documents/{target_document_id}/segments'
    segments_response = requests.get(segments_url, headers=headers, timeout=30)
    
    if segments_response.status_code != 200:
        return None
    
    segments_data = segments_response.json()
    segments = segments_data.get('data', [])
    
    # 合并所有segment的内容
    full_content = []
    for segment in sorted(segments, key=lambda x: x.get('position', 0)):
        content = segment.get('content', '')
        if content.strip():
            full_content.append(content.strip())
    
    return '\n\n'.join(full_content)

def enhanced_retrieve(query: str, top_k: int = 3) -> str:
    """
    增强检索流程：
    1. 从abstract dataset中检索
    2. 提取case_id
    3. 从full text dataset中获取完整文档
    """
    print(f"🔍 开始增强检索: {query}")
    print("="*60)
    
    # 步骤1: 从abstract dataset检索
    print("📝 步骤1: 从abstract dataset检索相关内容...")
    abstract_results = retrieve_from_abstract_dataset(query, top_k)
    
    records = abstract_results.get('records', [])
    if not records:
        return f"❌ 在abstract dataset中未找到相关内容"
    
    print(f"✅ 在abstract dataset中找到 {len(records)} 条相关记录")
    
    # 步骤2: 提取case_id并获取完整文档
    full_documents = []
    processed_case_ids = set()
    
    for i, record in enumerate(records, 1):
        # 修正：case_id在document.doc_metadata中，不是在segment.metadata中
        document = record.get('segment', {}).get('document', {})
        doc_metadata = document.get('doc_metadata', {})
        case_id = doc_metadata.get('case_id')
        
        if not case_id or case_id in processed_case_ids:
            continue
            
        processed_case_ids.add(case_id)
        
        print(f"📄 步骤2.{i}: 获取case_id={case_id}的完整文档...")
        full_content = get_document_by_case_id(case_id)
        
        if full_content:
            full_documents.append({
                'case_id': case_id,
                'content': full_content,
                'abstract_snippet': record.get('segment', {}).get('content', '')[:200] + '...'
            })
            print(f"✅ 成功获取case_id={case_id}的完整文档 ({len(full_content)}字符)")
        else:
            print(f"❌ 未能获取case_id={case_id}的完整文档")
    
    # 步骤3: 格式化输出结果
    if not full_documents:
        return "❌ 未能获取到任何完整文档"
    
    result_lines = [
        f"📊 检索结果总结:",
        f"- 查询: {query}",
        f"- Abstract检索到: {len(records)} 条记录",
        f"- 获取完整文档: {len(full_documents)} 个",
        "",
        "📚 完整文档内容:"
    ]
    
    for i, doc in enumerate(full_documents, 1):
        result_lines.extend([
            f"\n{'='*50}",
            f"文档 {i}: Case ID = {doc['case_id']}",
            f"{'='*50}",
            f"摘要片段: {doc['abstract_snippet']}",
            f"\n完整内容:",
            doc['content']
        ])
    
    return '\n'.join(result_lines)

if __name__ == "__main__":
    # 测试增强检索功能
    test_query = "朴愿有机农耕科普园"
    print(f"测试查询: {test_query}")
    print(f"{'='*80}")
    result = enhanced_retrieve(test_query)
    print(result)