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
    system = cl.user_session.get("rag_system")
    chat_history = cl.user_session.get("chat_history", [])

    if not system:
        await cl.Message(content="系统尚未初始化完成，请刷新页面重试。").send()
        return

    # 准备最终的回复消息对象
    response_msg = cl.Message(content="")
    full_response = ""

    try:
        # ---------------------------------------------------------
        # 步骤 1：意图分析与查询重写
        # ---------------------------------------------------------
        async with cl.Step(name="🤔 正在分析您的需求...") as step_analyze:
            # 使用 to_thread 避免同步请求阻塞 UI 动画
            route_type = await asyncio.to_thread(system.generation_module.query_router, message.content)

            if route_type == 'list':
                rewritten_query = message.content
                step_analyze.output = f"意图识别：菜品推荐/列表查询"
            else:
                step_analyze.output = "结合上下文智能分析查询意图中..."
                rewritten_query = await asyncio.to_thread(
                    system.generation_module.query_rewrite, message.content, chat_history
                )
                if rewritten_query != message.content:
                    step_analyze.output = f"意图识别：具体制作查询\n上下文补全：'{message.content}' ➡️ '{rewritten_query}'"
                else:
                    step_analyze.output = f"意图识别：具体制作查询"

        # ---------------------------------------------------------
        # 步骤 2：向量库检索
        # ---------------------------------------------------------
        async with cl.Step(name="🔍 正在翻阅菜谱库...") as step_retrieval:
            filters = system._extract_filters_from_query(message.content)

            if filters:
                step_retrieval.output = f"识别到过滤条件: {filters}\n正在进行精确检索..."
                relevant_chunks = await asyncio.to_thread(
                    system.retrieval_module.metadata_filtered_search, rewritten_query, filters, top_k=system.config.top_k
                )
            else:
                step_retrieval.output = "正在进行混合语义检索..."
                relevant_chunks = await asyncio.to_thread(
                    system.retrieval_module.hybrid_search, rewritten_query, top_k=system.config.top_k
                )

            if not relevant_chunks:
                step_retrieval.output = "❌ 未找到相关食谱。"
                await cl.Message(content="抱歉，厨房里没有找到相关的食谱信息。要不要尝试换个菜名或关键词？").send()
                return

            # 获取完整文档
            relevant_docs = await asyncio.to_thread(system.data_module.get_parent_documents, relevant_chunks)
            doc_names = [doc.metadata.get('dish_name', '未知菜品') for doc in relevant_docs]
            step_retrieval.output = f"✅ 成功找到参考菜谱：\n" + "\n".join([f"- {name}" for name in doc_names])

        # ---------------------------------------------------------
        # 步骤 3：模型生成回答
        # ---------------------------------------------------------
        async with cl.Step(name="👨‍🍳 正在为您烹饪回答...") as step_generate:
            step_generate.output = "正在整理制作步骤和技巧，请稍候..."

            # 发送空消息，为流式输出做准备
            await response_msg.send()

            if route_type == 'list':
                # 列表查询直接返回
                response = await asyncio.to_thread(
                    system.generation_module.generate_list_answer, message.content, relevant_docs
                )
                full_response = response
                response_msg.content = full_response
                await response_msg.update()
            else:
                # 获取流式生成器
                if route_type == "detail":
                    generator = system.generation_module.generate_step_by_step_answer_stream(message.content, relevant_docs, chat_history)
                else:
                    generator = system.generation_module.generate_basic_answer_stream(message.content, relevant_docs, chat_history)

                # 遍历生成器，实现打字机效果
                # 把生成过程放到独立线程，通过 asyncio 获取，防止阻塞 UI
                def fetch_chunks():
                    return [chunk for chunk in generator]

                # 注意：为了让流式打印平滑，最好是迭代原生生成器。
                # 但因为底层的大模型调用是同步的，最安全的做法是将每次获取 chunk 的动作包裹起来
                for chunk in generator:
                    full_response += chunk
                    await response_msg.stream_token(chunk)

                await response_msg.update()

            step_generate.output = "✅ 回答生成完毕！"

        # ---------------------------------------------------------
        # 步骤 4：历史记录管理
        # ---------------------------------------------------------
        chat_history.append(("human", message.content))
        chat_history.append(("ai", full_response))

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        await cl.Message(content=f"❌ 厨房出现了一点小意外: {str(e)}").send()