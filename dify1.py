import requests # 导入 requests 库，用于发送 HTTP 请求
import json     # 导入 json 库，用于处理 JSON 数据
from datetime import datetime # 导入 datetime 模块，用于获取当前时间并格式化

# Dify 知识库检索 API 端点
# 请根据您的 Dify 部署和知识库 ID 调整此 URL
url = 'http://localhost/v1/datasets/d29a3ad4-cbaa-4adf-98b9-94d2d2da8660/retrieve'
# Dify API 密钥
# 这是一个数据集 API 密钥，用于访问特定的知识库。如果您需要调用其他 Dify 功能（如聊天），
# 可能需要使用应用 API 密钥，并调整相应的 API 端点。
api_key = 'dataset-nn9K2CMUXa9rSKLlNpMwmHU7'  

# 请求头信息
headers = {
    'Authorization': f'Bearer {api_key}', # 设置认证头，包含 API 密钥
    'Content-Type': 'application/json',   # 指定请求体为 JSON 格式
}

# 请求体数据
# 这是一个 Dify 知识库检索请求的示例数据。
# "query": "生态农业" 是您要查询的内容。
# "response_mode": "blocking" 表示同步等待响应。
# "user": "1" 是用户标识，可根据实际情况修改。
data = {
    "inputs": {}, # 输入参数，对于知识库检索通常为空字典
    "query": "生态农业", # 查询文本
    "response_mode": "blocking", # 响应模式：blocking (同步) 或 streaming (流式)
    "conversation_id": "", # 会话 ID，如果需要保持会话上下文，可以设置
    "user": "1", # 用户 ID，用于 Dify 统计和管理
}

# 定义一个函数，用于优美地格式化和打印搜索结果
def format_search_results(response_data):
    """优美地格式化搜索结果"""
    print("=" * 80) # 打印分隔线
    # 打印标题，包含当前查询时间
    print(f"🔍 知识库检索结果 - 查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80) # 打印分隔线
    
    # 从响应数据中获取查询内容，如果不存在则默认为 '未知查询'
    query_content = response_data.get('query', {}).get('content', '未知查询')
    print(f"📝 查询内容: {query_content}") # 打印查询内容
    
    # 从响应数据中获取检索到的记录列表，如果不存在则默认为空列表
    records = response_data.get('records', [])
    print(f"📊 检索到 {len(records)} 条相关记录\n") # 打印检索到的记录数量
    
    # 遍历每条检索到的记录并打印其详细信息
    for i, record in enumerate(records, 1):
        # 获取片段信息，如果不存在则默认为空字典
        segment = record.get('segment', {})
        # 获取片段 ID，如果不存在则默认为 '未知ID'
        segment_id = segment.get('id', '未知ID')
        # 获取片段在文档中的位置，如果不存在则默认为 '未知位置'
        position = segment.get('position', '未知位置')
        # 获取片段内容，如果不存在则默认为 '无内容'
        content = segment.get('content', '无内容')
        
        print(f"📄 记录 {i}:") # 打印记录序号
        print(f"   🆔 片段ID: {segment_id}") # 打印片段 ID
        print(f"   📍 位置: {position}") # 打印片段位置
        # 打印内容预览，只显示前200个字符，如果内容超过200字符则添加 '...' 
        print(f"   📖 内容预览: {content[:200]}{'...' if len(content) > 200 else ''}")
        print("-" * 60) # 打印记录分隔线
    
    print("✅ 检索完成!") # 打印检索完成提示

# 主程序逻辑：发送请求并处理响应
try:
    print("🚀 正在向 Dify 知识库发送查询请求...") # 打印请求发送提示
    # 发送 POST 请求到 Dify API
    response = requests.post(url, headers=headers, json=data)
    
    # 检查 HTTP 响应状态码是否为 200 (成功)
    if response.status_code == 200:
        try:
            # 尝试将响应内容解析为 JSON
            json_response = response.json()
            # 调用格式化函数打印结果
            format_search_results(json_response)
            
            # 可选功能：询问用户是否保存完整结果到文件
            save_to_file = input("\n💾 是否保存完整结果到文件? (y/n): ").lower().strip()
                
        except requests.exceptions.JSONDecodeError:
            # 如果响应不是有效的 JSON 格式，捕获 JSONDecodeError
            print("❌ 响应不是有效的 JSON 格式") # 打印错误信息
            print(f"原始响应: {response.text}") # 打印原始响应内容，以便调试
    else:
        # 如果 HTTP 状态码不是 200，打印请求失败信息
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}") # 打印服务器返回的错误信息
        
except requests.exceptions.ConnectionError:
    # 捕获连接错误，例如服务器未运行或网络问题
    print("❌ 连接错误：无法连接到服务器。请确保 Dify 服务器正在运行。")
except requests.exceptions.RequestException as e:
    # 捕获其他所有 requests 库相关的异常
    print(f"❌ 请求异常: {e}") # 打印具体的请求异常信息
