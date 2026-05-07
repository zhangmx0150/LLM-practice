from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

# ==========================================
# 第二步：加载本地索引
# ==========================================
print(f"⏳ 正在从 {persist_path} 加载索引...")
# 指定之前的存储目录
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
# 从存储上下文中恢复索引对象
index = load_index_from_storage(storage_context)
print("✅ 索引加载成功！\n")

# ==========================================
# 第三步：执行相似性搜索
# ==========================================
# 将索引转换为“检索器 (Retriever)”，并设置召回最相似的 2 个结果
retriever = index.as_retriever(similarity_top_k=1)

# 用户提问
query_text = "LlamaIndex 是什么？"
print(f"🔍 搜索问题: '{query_text}'\n")

# 执行底层向量检索 (这一步只查相关文档，不调用大模型生成答案)
nodes = retriever.retrieve(query_text)

# ==========================================
# 第四步：打印检索结果
# ==========================================
print("📝 检索到的相似片段:")
for i, node in enumerate(nodes, 1):
    # node.score 代表余弦相似度得分，通常越接近 1 越相关
    print(f"--- 结果 {i} (相似度: {node.score:.4f}) ---")
    print(node.text)
    print()
