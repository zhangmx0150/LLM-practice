"""
私有知识库问答系统
支持 PDF、TXT、DOCX 文档，自动向量化存储
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 加载环境变量
load_dotenv()

class PrivateKnowledgeBase:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = DashScopeEmbeddings()
        self.vectordb = None
        self.qa_chain = None

        # 初始化 LLM
        self.llm = ChatTongyi(
            model="qwen-plus",
            temperature=0.3,   # 降低随机性，提高准确性
            max_tokens=1000,
            streaming=True
        )

    def load_documents(self, folder_path):
        """加载文件夹中的所有文档"""
        documents = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif file.endswith('.txt'):
                        loader = TextLoader(file_path)
                    elif file.endswith(('.docx', '.doc')):
                        loader = UnstructuredWordDocumentLoader(file_path)
                    else:
                        continue

                    documents.extend(loader.load())
                    print(f"✅ 已加载: {file}")
                except Exception as e:
                    print(f"❌ 加载失败 {file}: {e}")

        return documents

    def split_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """将文档分割成小块"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return text_splitter.split_documents(documents)

    def build_vectorstore(self, chunks):
        """构建向量数据库"""
        print("🔄 正在生成向量嵌入...")
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ 向量数据库已保存到: {self.persist_directory}")
        return self.vectordb

    def load_vectorstore(self):
        """加载已有的向量数据库"""
        if os.path.exists(self.persist_directory):
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("✅ 向量数据库加载成功")
            return True
        return False

    def setup_qa_chain(self, top_k=4):
        """设置问答链"""
        if not self.vectordb:
            raise ValueError("请先构建或加载向量数据库")

        retriever = self.vectordb.as_retriever(search_kwargs={"k": top_k})

        # 1. 定义 Prompt 模板
        template = """使用以下检索到的上下文来回答问题。
        如果你不知道答案，就说你不知道，不要试图编造答案。
        
        上下文: {context}
        
        问题: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 2. 定义文档格式化函数，将多个 Document 对象拼接成纯文本供大模型阅读
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 3. 构建子链：负责将格式化后的文本和问题塞给大模型并输出纯字符串
        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt
                | self.llm
                | StrOutputParser()
        )

        # 4. 构建主链：使用 RunnableParallel 并行处理，保留原始检索文档（为了前端展示来源）
        self.qa_chain = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        print("✅ 问答系统已就绪")

    def ask(self, question):
        """向知识库提问"""
        if not self.qa_chain:
            raise ValueError("请先设置问答链")

        sources = []
        # 使用 stream 替代 invoke
        for chunk in self.qa_chain.stream(question):
            # 1. 拦截并保存检索到的参考文档 (通常在流的第一个 chunk 中完整返回)
            if "context" in chunk and not sources:
                sources = [doc.page_content[:200] + "..." for doc in chunk["context"]]

            # 2. 拦截并 yield 大模型生成的文本片段
            if "answer" in chunk:
                # 每次 yield 产出一个文字块和来源列表
                yield chunk["answer"], sources


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 初始化知识库
    kb = PrivateKnowledgeBase(persist_directory="./vector_db")

    # 方式一：从文档构建（首次使用）
    # print("\n📂 开始构建知识库...")
    # docs = kb.load_documents("./documents")  # 放入你的文档目录
    # chunks = kb.split_documents(docs)
    # kb.build_vectorstore(chunks)
    # kb.setup_qa_chain()

    # 方式二：直接加载已有知识库
    kb.load_vectorstore()
    kb.setup_qa_chain()

    # 开始问答
    print("\n💬 知识库问答系统已启动，输入 'exit' 退出\n")

    while True:
        question = input("请输入问题: ")
        if question.lower() == "exit":
            break

        # result = kb.ask(question)
        print("\n📖 回答:\n", end="")
        final_sources = []
        # 遍历流式返回的生成器
        for content_chunk, sources in kb.ask(question):
            # 实时打印文本片段，不换行，并强制刷新缓冲区
            print(content_chunk, end="", flush=True)
            # 持续更新来源引用，循环结束时即为完整的引用列表
            final_sources = sources

        print("\n") # 回答打印完毕后换行

        # 如果需要打印来源，把这几行取消注释即可
        # print(f"📌 参考来源 ({len(final_sources)} 条):")
        # for i, src in enumerate(final_sources, 1):
        #     print(f"  {i}. {src}\n")