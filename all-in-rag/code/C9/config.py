"""
基于图数据库的RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "qwen3.6-plus"

    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度

    # 性能优化配置
    cache_enabled: bool = True  # 是否启用查询、检索、生成等运行期缓存
    cache_ttl_seconds: int = 3600  # 通用缓存过期时间
    cache_max_size: int = 256  # 通用缓存最大条目数
    answer_cache_ttl_seconds: int = 1800  # 答案缓存过期时间，避免过旧回答长期复用
    answer_cache_max_size: int = 128  # 答案缓存最大条目数
    enable_parallel_retrieval: bool = True  # 是否并行执行独立检索分支
    embedding_device: str = "auto"  # auto/cpu/cuda；auto 会优先使用可用 GPU

    def __post_init__(self):
        """初始化后规范化配置值，避免非法参数影响缓存和检索。"""
        self.cache_ttl_seconds = max(1, int(self.cache_ttl_seconds))
        self.cache_max_size = max(1, int(self.cache_max_size))
        self.answer_cache_ttl_seconds = max(1, int(self.answer_cache_ttl_seconds))
        self.answer_cache_max_size = max(1, int(self.answer_cache_max_size))
        self.embedding_device = (self.embedding_device or "auto").lower()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,

            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth,
            'cache_enabled': self.cache_enabled,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'cache_max_size': self.cache_max_size,
            'answer_cache_ttl_seconds': self.answer_cache_ttl_seconds,
            'answer_cache_max_size': self.answer_cache_max_size,
            'enable_parallel_retrieval': self.enable_parallel_retrieval,
            'embedding_device': self.embedding_device
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig()
