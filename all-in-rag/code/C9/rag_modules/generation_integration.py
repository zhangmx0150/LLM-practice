"""
生成集成模块
"""

import logging
import os
import time
from typing import List

from openai import OpenAI
from langchain_core.documents import Document
from .cache_utils import TTLCache, document_fingerprint, make_cache_key

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""

    def __init__(
        self,
        model_name: str = "qwen3.6-plus",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        cache_enabled: bool = True,
        answer_cache_max_size: int = 128,
        answer_cache_ttl_seconds: int = 1800,
    ):
        """
        初始化生成集成模块，创建 DashScope/OpenAI 兼容客户端和答案缓存。
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.answer_cache = TTLCache(
            max_size=answer_cache_max_size,
            ttl_seconds=answer_cache_ttl_seconds,
            enabled=cache_enabled,
        )
        
        # 初始化OpenAI客户端
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        logger.info(f"生成模块初始化完成，模型: {model_name}")

    def _build_context(self, documents: List[Document]) -> str:
        """把检索到的文档整理成提示词上下文，保留原有层级标记逻辑。"""
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if not content:
                continue

            level = doc.metadata.get("retrieval_level", "")
            if level:
                context_parts.append(f"[{level.upper()}] {content}")
            else:
                context_parts.append(content)

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, documents: List[Document]) -> str:
        """根据问题和检索文档构建与原逻辑一致的烹饪问答提示词。"""
        context = self._build_context(documents)
        return f"""
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

    def _answer_cache_key(self, question: str, documents: List[Document]) -> str:
        """根据问题、文档指纹和生成参数生成答案缓存键。"""
        document_keys = [document_fingerprint(doc) for doc in documents]
        return make_cache_key(
            "adaptive_answer",
            self.model_name,
            self.temperature,
            self.max_tokens,
            question,
            document_keys,
        )

    def clear_cache(self) -> None:
        """清空答案缓存，适用于知识库重建或希望强制重新生成的场景。"""
        self.answer_cache.clear()

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """
        智能统一答案生成；命中缓存时直接返回，未命中时保持原有 LLM 生成流程。
        """
        cache_key = self._answer_cache_key(question, documents)
        cached_answer = self.answer_cache.get(cache_key)
        if cached_answer:
            logger.info("答案生成命中缓存")
            return cached_answer

        prompt = self._build_prompt(question, documents)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content.strip()
            self.answer_cache.set(cache_key, answer)
            return answer
            
        except Exception as e:
            logger.error(f"LightRAG答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 3):
        """
        LightRAG风格的流式答案生成；完整生成成功后写入缓存，重复问题可直接流式吐出缓存文本。
        """
        cache_key = self._answer_cache_key(question, documents)
        cached_answer = self.answer_cache.get(cache_key)
        if cached_answer:
            logger.info("流式答案命中缓存")
            yield cached_answer
            return

        prompt = self._build_prompt(question, documents)

        for attempt in range(max_retries):
            chunk_yielded = False  # 记录是否已经成功输出过字符
            answer_parts = []
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
                        answer_parts.append(content)
                        yield content
                        chunk_yielded = True

                # 如果顺利走完循环，说明生成完整结束，此时缓存完整答案。
                if answer_parts:
                    self.answer_cache.set(cache_key, "".join(answer_parts).strip())
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
