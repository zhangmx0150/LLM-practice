import chainlit as cl
import asyncio
from main import RecipeRAGSystem

# ==========================================
# 🌟 1. 全局单例：让所有刷新动作和所有用户共享这一个知识库实例
# ==========================================
global_rag_system = None

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
    # 声明使用外部的全局变量
    global global_rag_system

    # 为当前用户开启一块专属的“记忆空间”
    cl.user_session.set("chat_history", [])

    msg = cl.Message(content="👩‍🍳 正在检查厨房状态，请稍候...")
    await msg.send()

    try:
        # 如果全局变量为空，说明是服务器刚启动，真正执行加载
        if global_rag_system is None:
            msg.content = "👩‍🍳 厨房首次开火（初始化大模型与知识库），这可能需要几十秒，请稍候..."
            await msg.update()
            # 在后台线程加载，防止阻塞
            global_rag_system = await asyncio.to_thread(init_system)

        # 如果全局变量不为空（比如你只是刷新了网页），直接复用
        else:
            print("⚡ 检测到全局知识库已存在，直接复用内存实例！")

        # 将全局实例挂载到当前用户的会话中，供后续问答使用
        cl.user_session.set("rag_system", global_rag_system)

        # 更新欢迎语
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
    # 取出当前用户的历史记录
    chat_history = cl.user_session.get("chat_history", [])

    if not system:
        await cl.Message(content="系统尚未初始化完成，请刷新页面重试。").send()
        return

    # 准备一个空消息对象，用于一会的流式输出
    response_msg = cl.Message(content="")
    full_response = "" # 用于拼接大模型最终的回答以存入记忆

    try:
        # 把历史记录传给后端的 ask_question，开启流式输出
        response = system.ask_question(message.content, stream=True, chat_history=chat_history)

        # 兼容性处理：如果返回的是纯字符串（比如找不到菜谱时的提示）
        if isinstance(response, str):
            full_response = response
            response_msg.content = full_response
            await response_msg.send()

        # 如果返回的是流式生成器（详情步骤输出）
        else:
            # Chainlit 必须先 send() 一个空消息，然后才能往里面塞数据
            await response_msg.send()

            # 遍历生成器，原生支持流式打印
            for chunk in response:
                full_response += chunk
                await response_msg.stream_token(chunk)

            # 传输完毕，更新最终状态
            await response_msg.update()

        # 核心逻辑：一轮对话结束后，将问答双双追加进记忆里
        # 使用 ("角色", "内容") 的元组格式，完美适配 LangChain 的 MessagesPlaceholder
        chat_history.append(("human", message.content))
        chat_history.append(("ai", full_response))

        # 限制记忆长度（例如只保留最近的 5 轮对话 / 10条消息，防止Token超载报错）
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        # 重新存回 session
        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        await cl.Message(content=f"❌ 处理问题时出错: {str(e)}").send()