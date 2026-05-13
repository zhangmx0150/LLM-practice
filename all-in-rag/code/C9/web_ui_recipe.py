import chainlit as cl
import asyncio
from main import AdvancedGraphRAGSystem

# ==========================================
# 🌟 全局单例：共享图数据库与向量库连接，实现秒开
# ==========================================
global_rag_system = None

def init_system():
    """包装初始化过程，以便在后台线程中运行"""
    system = AdvancedGraphRAGSystem()
    system.initialize_system()
    system.build_knowledge_base()
    return system

@cl.on_chat_start
async def on_chat_start():
    global global_rag_system

    # 开启用户的记忆空间
    cl.user_session.set("chat_history", [])

    msg = cl.Message(content="👩‍🍳 正在连接 Milvus 向量库与 Neo4j 图数据库，请稍候...")
    await msg.send()

    try:
        if global_rag_system is None:
            msg.content = "👩‍🍳 首次开火（初始化双引擎检索系统），这可能需要几十秒，请稍候..."
            await msg.update()
            # 后台线程加载，防止卡死 UI
            global_rag_system = await asyncio.to_thread(init_system)
        else:
            print("⚡ 检测到全局知识库已存在，直接复用内存实例！")

        cl.user_session.set("rag_system", global_rag_system)

        msg.content = "✅ **高级图谱厨房准备就绪！**\n\n您好！我是接入了 GraphRAG 的智能厨师。无论是简单的做法，还是复杂的食材替代与推理，我都能解答。\n\n*(💡 例如：包含茄子但不需要油炸的四星素菜有哪些？ / 鱼香肉丝可以用什么替代木耳？)*"
        await msg.update()

    except Exception as e:
        msg.content = f"❌ 系统初始化失败: {str(e)}"
        await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    system = cl.user_session.get("rag_system")
    chat_history = cl.user_session.get("chat_history", [])

    if not system:
        await cl.Message(content="系统尚未初始化完成，请刷新页面重试。").send()
        return

    response_msg = cl.Message(content="")
    full_response = ""

    try:
        # 🌟 核心修复 1：包装一个内部函数，用于在后台线程执行耗时的检索和路由
        def fetch_answer():
            return system.ask_question_with_routing(
                question=message.content,
                stream=True,
                explain_routing=True
            )

        # 使用 asyncio.to_thread 将卡顿的操作甩给后台，主线程继续保持和网页的心跳！
        response, analysis = await asyncio.to_thread(fetch_answer)

        # 1. 兼容纯字符串返回
        if isinstance(response, str):
            full_response = response
            await response_msg.stream_token(full_response)

        # 2. 处理流式生成器
        else:
            await response_msg.send()
            for chunk in response:
                full_response += chunk
                await response_msg.stream_token(chunk)
                # 🌟 核心修复 2：在同步的 for 循环中，强制让出极短的 CPU 时间给心跳程序
                await asyncio.sleep(0)

        await response_msg.update()

        # 记录历史
        chat_history.append(("human", message.content))
        chat_history.append(("ai", full_response))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        await cl.Message(content=f"❌ 处理问题时出错: {str(e)}").send()