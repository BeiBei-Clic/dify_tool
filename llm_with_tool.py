import os
import requests
import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量读取配置
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
DIFY_BASE_URL = os.getenv('DIFY_BASE_URL', 'http://localhost')
DIFY_DATASET_ID = os.getenv('DIFY_DATASET_ID')
DIFY_API_KEY = os.getenv('DIFY_API_KEY')

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
    # 创建LLM实例
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.7
    )
    
    # 创建ReAct Agent
    agent = create_react_agent(
        model=llm,
        tools=[dify_retrieve],
        prompt="你是一个有用的AI助手，可以使用工具来检索信息并回答用户问题。请基于检索到的信息给出详细、有用的回答。"
    )
    
    # 测试查询
    test_query = "请帮我查询生态农业相关的信息"
    print(f"🤖 向Agent发送查询: {test_query}")
    print("=" * 60)
    
    # 使用 stream 模式实时查看思考过程
    print("🧠 大模型思考过程:")
    print("-" * 40)
    
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": test_query}]
    }):
        # 打印每一步的思考过程
        for node_name, node_data in chunk.items():
            if "messages" in node_data:
                latest_message = node_data["messages"][-1]
                
                if hasattr(latest_message, 'type'):
                    if latest_message.type == "ai":
                        print(f"🤖 [{node_name}] AI思考: {latest_message.content}")
                        
                        # 如果有工具调用，显示工具调用信息
                        if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                            print(f"🔧 [{node_name}] 准备调用工具:")
                            for tool_call in latest_message.tool_calls:
                                print(f"   - 工具: {tool_call['name']}")
                                print(f"   - 参数: {tool_call['args']}")
                                
                    elif latest_message.type == "tool":
                        print(f"⚡ [{node_name}] 工具执行结果:")
                        print(f"   {latest_message.content[:200]}...")  # 只显示前200字符
                        
                print("-" * 40)
    
    print("\n✅ 思考过程完成")
    # 调用Agent
    response = agent.invoke({
        "messages": [{"role": "user", "content": test_query}]
    })
    
    print("📋 完整对话过程:")
    print("-" * 40)
    
    # 遍历所有消息，展示完整的对话流程
    for i, message in enumerate(response["messages"], 1):
        if hasattr(message, 'content') and hasattr(message, 'type'):
            if message.type == "human":
                print(f"👤 用户 ({i}): {message.content}")
            elif message.type == "ai":
                print(f"🤖 AI ({i}): {message.content}")
                # 如果有工具调用，显示工具调用信息
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print(f"🔧 工具调用: {len(message.tool_calls)} 个")
                    for tool_call in message.tool_calls:
                        print(f"   - 工具: {tool_call['name']}")
                        print(f"   - 参数: {tool_call['args']}")
            elif message.type == "tool":
                print(f"⚡ 工具结果 ({i}):")
                print(f"   {message.content}")
        print("-" * 40)
    
    # 提取最终回答
    final_message = response["messages"][-1]
    if hasattr(final_message, 'content'):
        print("\n🎯 最终回答:")
        print(final_message.content)
    else:
        print("\n⚠️ 未找到最终回答")