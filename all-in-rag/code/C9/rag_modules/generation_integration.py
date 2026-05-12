"""
生成集成模块
"""

import logging
import os
import time
from typing import List

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""

    def __init__(self, model_name: str = "qwen3.6-plus", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化OpenAI客户端
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        logger.info(f"生成模块初始化完成，模型: {model_name}")

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """
        智能统一答案生成
        自动适应不同类型的查询，无需预先分类
        """
        # 构建上下文
        context_parts = []
        
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                # 添加检索层级信息（如果有的话）
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)
        
        context = "\n\n".join(context_parts)
        
        # LightRAG风格的统一提示词
        prompt = f"""
        作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

        检索到的相关信息：
        {context}

        用户问题：{question}

        请提供准确、实用的回答。根据问题的性质：
        - 如果是询问多个菜品，请提供清晰的列表
        - 如果是询问具体制作方法，请提供详细步骤
        - 如果是一般性咨询，请提供综合性回答

        回答：
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LightRAG答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 3):
        """
        LightRAG风格的流式答案生成（带重试机制）
        """
        # 构建上下文
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)

        context = "\n\n".join(context_parts)

        prompt = f"""
        作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

        检索到的相关信息：
        {context}

        用户问题：{question}

        请提供准确、实用的回答。根据问题的性质：
        - 如果是询问多个菜品，请提供清晰的列表
        - 如果是询问具体制作方法，请提供详细步骤
        - 如果是一般性咨询，请提供综合性回答

        回答：
        """

        for attempt in range(max_retries):
            chunk_yielded = False # 记录是否已经成功输出过字符
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )

                for chunk in response:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    if getattr(delta, 'content', None):
                        content = delta.content
                        yield content
                        chunk_yielded = True

                # 如果顺利走完循环，说明生成完美结束，退出函数
                return

            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")

                # 否则前端会收到两遍相同的开头文本（比如：你好！红烧肉...你好！红烧肉...）
                if chunk_yielded:
                    yield f"\n\n[⚠️ 网络波动，回答中断]"
                    return

                if attempt < max_retries - 1:
                    # 注意：在真实生产的异步 UI 中应尽量避免同步 sleep，
                    # 但经过上面的解析修复，99% 的情况下已经不会触发这里了。
                    time.sleep(2)
                    continue
                else:
                    logger.error("流式生成完全失败，尝试非流式后备方案")
                    try:
                        fallback_response = self.generate_adaptive_answer(question, documents)
                        yield fallback_response
                        return
                    except Exception as fallback_error:
                        yield f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                        return