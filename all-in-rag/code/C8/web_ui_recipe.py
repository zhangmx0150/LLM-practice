import chainlit as cl
import asyncio
from main import RecipeRAGSystem

# ==========================================
# 🌟 1. 系统初始化逻辑 (后台线程运行防阻塞)
# ==========================================
def init_system():
    """包装初始化过程，以便在单独的线程中运行"""
    system = RecipeRAGSystem()
    system.initialize_system()
    system.build_knowledge_base()
    return system

# ==========================================
# 🌟 2. 当用户打开聊天窗口（开始新会话）时触发
# ==========================================
@cl.on_chat_start
async def on_chat_start():
    # 发送一个加载提示（类似 Streamlit 的 st.spinner）
    msg = cl.Message(content="👩‍🍳 正在生火热锅（初始化知识库），这可能需要几十秒的时间，请稍候...")
    await msg.send()

    try:
        # 使用 asyncio.to_thread 将耗时的加载任务放到后台，不卡死界面
        system = await asyncio.to_thread(init_system)

        # 将初始化好的系统存入当前用户的 session 中（类似 st.session_state）
        cl.user_session.set("rag_system", system)

        # 更新刚才那条加载提示消息
        msg.content = "✅ 厨房准备就绪，可以开始点菜啦！\n\n您好！我是您的私人厨师助手。想了解哪道菜的做法，或者需要推荐菜单吗？\n\n*(💡 例如：宫保鸡丁怎么做？ / 推荐几个简单的素菜 / 鱼香肉丝需要什么食材？)*"
        await msg.update()

    except Exception as e:
        msg.content = f"❌ 系统初始化失败: {str(e)}"
        await msg.update()

# ==========================================
# 🌟 3. 当接收到用户的消息时触发
# ==========================================
@cl.on_message
async def on_message(message: cl.Message):
    # 从 session 中取出我们的 RAG 系统
    system = cl.user_session.get("rag_system")
    if not system:
        await cl.Message(content="系统尚未初始化完成，请刷新页面重试。").send()
        return

    # 准备一个空消息对象，用于一会的流式输出
    response_msg = cl.Message(content="")

    try:
        # 调用你的主程序方法，开启流式输出
        response = system.ask_question(message.content, stream=True)

        # 兼容性处理：如果返回的是纯字符串（比如找不到菜谱时的提示）
        if isinstance(response, str):
            response_msg.content = response
            await response_msg.send()

        # 如果返回的是流式生成器（详情步骤输出）
        else:
            # Chainlit 必须先 send() 一个空消息，然后才能往里面塞数据
            await response_msg.send()

            # 遍历生成器，原生支持流式打印，不需要模拟“▌”光标
            for chunk in response:
                await response_msg.stream_token(chunk)

            # 传输完毕，更新最终状态
            await response_msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ 处理问题时出错: {str(e)}").send()