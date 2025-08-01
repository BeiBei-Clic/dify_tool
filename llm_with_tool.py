import os
import requests
import asyncio
import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from dify2 import dify_retrieve

# 加载.env文件
load_dotenv()

# 从环境变量读取配置
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

async def stream_with_token_output():
    """使用 astream_events 实现逐 token 流式输出"""
    
    # 创建LLM实例，必须设置 streaming=True
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.7,
        streaming=True  # 关键：启用流式输出
    )
    
    # 创建ReAct Agent
    agent = create_react_agent(
        model=llm,
        tools=[dify_retrieve],
        prompt="你是一个有用的AI助手，可以使用工具来检索信息并回答用户问题。请基于检索到的信息给出详细、有用的回答。"
    )
    
    test_query = "请帮我查询生态农业相关的信息"
    print(f"🤖 向Agent发送查询: {test_query}")
    print("=" * 60)
    
    # 使用 astream_events 获取每个事件
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": test_query}]},
        version="v1"  # 使用 v1 版本的 API
    ):
        event_type = event.get("event")
        event_name = event.get("name", "")
        
        # 捕获 LLM 开始思考
        if event_type == "on_chat_model_start":
            print(f"\n🧠 [{event_name}] 开始思考...")
            
        # 捕获每个 token 的输出
        elif event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", {})
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                
        # 捕获 LLM 完成思考
        elif event_type == "on_chat_model_end":
            print(f"\n✅ [{event_name}] 思考完成")
            
        # 捕获工具调用开始
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event.get("data", {}).get("input", {})
            print(f"\n🔧 [{tool_name}] 开始执行工具")
            print(f"   参数: {json.dumps(tool_input, ensure_ascii=False)}")
            
        # 捕获工具执行结果
        elif event_type == "on_tool_end":
            tool_name = event.get("name", "")
            tool_output = event.get("data", {}).get("output", "")
            print(f"\n⚡ [{tool_name}] 工具执行完成")
            
            # 正确处理不同类型的输出
            if hasattr(tool_output, 'content'):
                output_content = tool_output.content
            elif isinstance(tool_output, str):
                output_content = tool_output
            else:
                output_content = str(tool_output)
                
            # 显示结果的前100个字符
            if len(output_content) > 100:
                print(f"   结果: {output_content[:100]}...")
            else:
                print(f"   结果: {output_content}")

if __name__ == "__main__":
    asyncio.run(stream_with_token_output())