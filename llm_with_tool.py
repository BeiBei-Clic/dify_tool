"""
LLM with Tool - 使用 LangChain 让大模型调用 Dify 知识库工具

这个程序演示了如何使用 LangChain 框架让大语言模型调用自定义工具。
参考 LangChain 官方文档的 use context 模式。
"""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# 加载 .env 文件中的环境变量
load_dotenv()

# 导入 dify2 中定义的知识库检索工具
from dify2 import dify_retrieve

# 从 .env 文件获取 OpenRouter 配置
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def print_beautiful_header():
    """打印美化的头部信息"""
    print("🤖 " + "=" * 60)
    print("🌟 LLM with Dify Tool - 智能农业知识助手")
    print("🤖 " + "=" * 60)
    print(f"🔧 使用 OpenRouter API")
    print(f"📡 Base URL: {openrouter_base_url}")
    print("🤖 " + "=" * 60)

def print_user_question(question):
    """美化打印用户问题"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n🙋 [{timestamp}] 用户提问:")
    print(f"   💬 {question}")
    print("   " + "─" * 50)

def print_tool_call_info(tool_calls):
    """美化打印工具调用信息"""
    if tool_calls:
        print(f"\n🔧 工具调用:")
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('args', {})
            print(f"   📋 工具 {i}: {tool_name}")
            print(f"   📝 参数: {tool_args}")
        print("   ⏳ 正在执行工具...")

def print_tool_result(content):
    """美化打印工具执行结果"""
    print(f"\n📊 工具执行结果:")
    print("   " + "─" * 45)
    
    # 格式化工具结果
    lines = content.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith('📝 查询:'):
                print(f"   🔍 {line}")
            elif line.startswith('📊 检索到'):
                print(f"   📈 {line}")
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                print(f"   📄 {line}")
            else:
                print(f"   {line}")
    print("   " + "─" * 45)

def print_final_answer(content):
    """美化打印最终回答"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n🤖 [{timestamp}] 智能助手回答:")
    print("   " + "═" * 50)
    
    # 格式化回答内容
    paragraphs = content.split('\n\n')
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            lines = paragraph.split('\n')
            for line in lines:
                if line.strip():
                    # 检测列表项
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '*', '-')):
                        print(f"   📌 {line.strip()}")
                    # 检测标题
                    elif '**' in line:
                        title = line.strip().replace('**', '')
                        print(f"   🎯 {title}")
                    else:
                        print(f"   {line}")
            
            # 段落间添加空行
            if i < len(paragraphs) - 1:
                print()
    
    print("   " + "═" * 50)

def print_summary(messages):
    """打印对话总结"""
    tool_calls_count = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_count += len(msg.tool_calls)
    
    print(f"\n💡 对话总结:")
    print(f"   ✅ 问题已成功处理")
    print(f"   🔧 工具调用次数: {tool_calls_count}")
    print(f"   🤖 已生成智能回答")
    print("   " + "─" * 30)

def beautify_result(result):
    """美化显示结果"""
    if 'messages' in result:
        messages = result['messages']
        
        # 找到用户问题
        user_question = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        if user_question:
            print_user_question(user_question)
        
        # 处理每条消息
        for msg in messages:
            if isinstance(msg, AIMessage):
                # 检查是否有工具调用
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print_tool_call_info(msg.tool_calls)
                # 检查是否有最终回答
                elif msg.content and msg.content.strip():
                    print_final_answer(msg.content)
            
            elif isinstance(msg, ToolMessage):
                # 显示工具执行结果
                print_tool_result(msg.content)
        
        # 打印总结
        print_summary(messages)
    
    else:
        # 如果结果格式不是预期的，直接打印
        print(f"\n📋 原始结果:")
        print(result)

# 打印美化的头部
print_beautiful_header()

# 创建 ChatOpenAI 实例，配置 OpenRouter
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    temperature=0.1,
    api_key=openrouter_api_key,
    base_url=openrouter_base_url,
)
    
agent = create_react_agent(
    model=llm,
    tools=[dify_retrieve]
)

if __name__=="__main__":
    # 可以修改这里的问题进行测试
    question = "如何实现乡村振兴？"
    
    print(f"\n🧠 大模型思考中...")
    print(f"   🔍 分析问题并决定是否需要调用工具...")
    
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    
    # 美化显示结果
    beautify_result(result)
