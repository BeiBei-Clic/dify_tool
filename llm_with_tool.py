import requests
import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Dify 配置常量
DIFY_BASE_URL = 'http://localhost'
DIFY_DATASET_ID = 'd29a3ad4-cbaa-4adf-98b9-94d2d2da8660'#上下册知识库id
DIFY_API_KEY = 'dataset-nn9K2CMUXa9rSKLlNpMwmHU7'

@tool
def dify_retrieve(query: str, user_id: str = "1", top_k: int = 5) -> str:
    """
    从 Dify 知识库中检索相关信息。
    
    这个工具可以根据用户的查询从 Dify 知识库中检索相关的文档片段。
    适用于需要获取特定领域知识、文档内容或回答基于知识库的问题。
    
    Args:
        query (str): 要检索的查询文本，例如 "生态农业的发展趋势"
        user_id (str, optional): 用户标识符，用于 Dify 统计。默认为 "1"
        top_k (int, optional): 返回的最大结果数量。默认为 5
        
    Returns:
        str: 格式化的检索结果，包含相关文档片段的内容
        
    Example:
        >>> result = dify_retrieve("生态农业")
        >>> print(result)
        检索到 3 条相关记录:
        1. [位置: 14] 生态农业是一种可持续的农业发展模式...
        2. [位置: 25] 生态农业注重环境保护和资源循环利用...
        3. [位置: 42] 现代生态农业技术包括有机种植、生物防治...
    """
    
    # 构建 API 端点 URL
    url = f'{DIFY_BASE_URL}/v1/datasets/{DIFY_DATASET_ID}/retrieve'
    
    # 设置请求头
    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json',
    }
    
    # 构建请求数据
    data = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "conversation_id": "",
        "user": user_id,
        "retrieval_setting": {
            "top_k": top_k
        }
    }
    
    try:
        # 发送 POST 请求到 Dify API
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        # 检查响应状态码
        if response.status_code == 200:
            try:
                # 解析 JSON 响应
                json_response = response.json()
                
                # 格式化检索结果
                return _format_retrieve_results(json_response, query)
                
            except json.JSONDecodeError:
                return f"❌ 解析响应失败：响应不是有效的 JSON 格式\n原始响应: {response.text[:500]}"
                
        else:
            return f"❌ 请求失败，状态码: {response.status_code}\n错误信息: {response.text[:500]}"
            
    except requests.exceptions.ConnectionError:
        return "❌ 连接错误：无法连接到 Dify 服务器。请确保服务器正在运行且网络连接正常。"
    except requests.exceptions.Timeout:
        return "❌ 请求超时：Dify 服务器响应时间过长，请稍后重试。"
    except requests.exceptions.RequestException as e:
        return f"❌ 请求异常: {str(e)}"


def _format_retrieve_results(response_data: Dict[str, Any], original_query: str) -> str:
    """
    格式化 Dify 检索结果为易读的字符串格式
    
    Args:
        response_data: Dify API 返回的 JSON 数据
        original_query: 原始查询文本
        
    Returns:
        str: 格式化后的结果字符串
    """
    
    # 获取查询内容
    query_content = response_data.get('query', {}).get('content', original_query)
    
    # 获取检索记录
    records = response_data.get('records', [])
    
    if not records:
        return f"📝 查询: {query_content}\n❌ 未找到相关记录，请尝试调整查询关键词。"
    
    # 构建结果字符串
    result_lines = [
        f"📝 查询: {query_content}",
        f"📊 检索到 {len(records)} 条相关记录:\n"
    ]
    
    for i, record in enumerate(records, 1):
        segment = record.get('segment', {})
        position = segment.get('position', '未知位置')
        content = segment.get('content', '无内容')
        
        # 清理内容：移除多余的换行符和空格
        cleaned_content = ' '.join(content.split())       
        result_lines.append(f"{i}. [位置: {position}] {cleaned_content}")
    
    return "\n".join(result_lines)

if __name__ == "__main__":
    # 运行测试
    """测试 dify_retrieve 工具函数"""
    print("🧪 测试 dify_retrieve 工具函数...")
    
    # 测试查询
    test_queries = [
        "生态农业",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"测试查询: {query}")
        print(f"{'='*60}")
        
        # 使用 invoke 方法而不是直接调用，避免弃用警告
        result = dify_retrieve.invoke({"query": query})
        print(result)