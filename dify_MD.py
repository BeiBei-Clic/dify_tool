import requests
import json
from typing import List, Dict, Any

# Dify 配置常量 - 参考 dify2.py 的配置
DIFY_BASE_URL = 'http://localhost'
ABSTRACT_DATASET_ID = '5a36aa21-0aa7-433e-bdab-72aa9931b543'  # 使用 dify_MD.py 中的知识库ID
FULL_TEXT_DATASET_ID="8e162a14-c930-40a9-8395-39502b58a45b"
DIFY_API_KEY = 'dataset-nn9K2CMUXa9rSKLlNpMwmHU7'

def get_all_documents() -> Dict[str, Any]:
    """
    获取知识库中所有文档的列表
    
    Returns:
        Dict: 包含文档列表的响应数据
    """
    url = f'{DIFY_BASE_URL}/v1/datasets/{DIFY_DATASET_ID}/documents'
    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"获取文档列表失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return {}
    except Exception as e:
        print(f"获取文档列表异常: {str(e)}")
        return {}

def display_documents_metadata():
    """
    显示所有文档的元数据信息
    """
    print("🔍 正在获取知识库中所有文档的元数据...")
    print("="*80)
    
    # 获取文档列表
    documents_data = get_all_documents()
    
    documents = documents_data.get('data', [])
    total = documents_data.get('total', 0)
    
    print(f"📊 知识库总共有 {total} 个文档")
    print("="*80)
        
    # 遍历每个文档，获取详细信息
    for i, doc in enumerate(documents, 1):
        print(doc)        
        print("-" * 60)

if __name__ == "__main__":
    # 显示所有文档的元数据
    display_documents_metadata()
