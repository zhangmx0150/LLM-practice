"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库 (支持持久化与增量更新)"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的向量索引和数据缓存
        vectorstore = self.index_module.load_index()
        cache_loaded = self.data_module.load_state_from_cache()

        # 检查是否有文件新增或修改
        updated_files = self.data_module.check_for_updates()

        if vectorstore is not None and cache_loaded and not updated_files:
            print("✅ 完美命中缓存！本地索引和数据无需更新，实现秒级启动。")
            chunks = self.data_module.chunks

        elif vectorstore is not None and cache_loaded and updated_files:
            print(f"🔄 发现 {len(updated_files)} 个新增或修改的文件，正在进行增量更新...")
            # 增量处理逻辑（这里为了简单和数据一致性，当有文件变更时，我们选择重新解析变更文件并追加）
            # 注：FAISS 删除旧向量较复杂，业内轻量级做法是增量追加。如果修改频繁，建议定期删掉 index 文件夹全量重建。
            self.data_module.load_documents() # 这里可以进一步优化为只 load updated_files
            chunks = self.data_module.chunk_documents()

            print("追加向量索引...")
            # 重新构建或追加（为了彻底避免重复，这里执行全量重建，配合缓存依然比以前快）
            vectorstore = self.index_module.build_vector_index(chunks)
            self.index_module.save_index()
            self.data_module.save_state_to_cache()

        else:
            print("⚠️ 未找到完整缓存或索引，开始全量构建新知识库...")
            self.data_module.file_hashes = {} # 清空旧指纹

            print("加载食谱文档...")
            self.data_module.load_documents()
            # 记录此时的所有文件指纹
            self.data_module.check_for_updates()

            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            print("构建并保存向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            self.index_module.save_index()

            # 保存数据缓存
            self.data_module.save_state_to_cache()

        # 2. 初始化检索优化模块 (BM25需要用到 chunks)
        print("初始化混合检索模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 3. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print("✅ 知识库就绪！")

    def ask_question(self, question: str, stream: bool = False, chat_history: List = None):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        chat_history = chat_history or []  # 默认空列表

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            rewritten_query = question
        else:
            print("🤖 结合上下文智能分析查询...")
            # 传入历史记录给重写器
            rewritten_query = self.generation_module.query_rewrite(question, chat_history)

        # 3. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题（仅取第一行）
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")
            if route_type == "detail":
                if stream:
                    # 传入历史记录给生成器
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs, chat_history)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                if stream:
                    # 传入历史记录给生成器
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs, chat_history)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品

        Args:
            category: 菜品分类
            query: 可选的额外查询条件

        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}

        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)

        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        return dish_names

    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

        return answer

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用尝尝咸淡RAG系统！")



def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()