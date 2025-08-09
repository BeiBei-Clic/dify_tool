import os
import asyncio
import json
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from dify_retrieval_enhanced import enhanced_retrieve

load_dotenv()

async def stream_with_token_output():
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
        temperature=0.7,
        streaming=True
    )
    
    agent = create_react_agent(
        model=llm,
        tools=[enhanced_retrieve],
        prompt="""
        你是一个有用的AI助手，可以使用工具来检索信息并回答用户问题。
        假如你连接数据库失败，就使用你自己的知识进行回答。
        """
    )
    
    test_query = "请帮我查询生态农业相关的信息"
    print(f"🤖 查询: {test_query}")
    print("=" * 60)
    
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": test_query}]},
        version="v1"
    ):
        event_type = event.get("event")
        
        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", {})
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event.get("data", {}).get("input", {})
            print(f"\n🔧 [{tool_name}] 执行中...")
            print(f"参数: {json.dumps(tool_input, ensure_ascii=False)}")
            
        elif event_type == "on_tool_end":
            tool_name = event.get("name", "")
            tool_output = event.get("data", {}).get("output", "")
            print(f"\n⚡ [{tool_name}] 完成")
            
            output_content = tool_output.content if hasattr(tool_output, 'content') else str(tool_output)
            print(f"结果: {output_content[:100]}..." if len(output_content) > 100 else f"结果: {output_content}")

if __name__ == "__main__":
    asyncio.run(stream_with_token_output())