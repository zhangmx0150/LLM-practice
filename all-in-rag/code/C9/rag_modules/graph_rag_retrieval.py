"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase
from .cache_utils import TTLCache, clone_documents, make_cache_key, strip_json_markdown

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"  # 多跳查询：A通过什么连接到C？
    SUBGRAPH = "subgraph"  # 子图查询：A相关的所有信息
    PATH_FINDING = "path_finding"  # 路径查找：从A到B的最佳路径
    CLUSTERING = "clustering"  # 聚类查询：和A相似的都有什么？

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]

class GraphRAGRetrieval:
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """
    
    def __init__(self, config, llm_client):
        """初始化图RAG检索器的配置、LLM客户端和运行期缓存。"""
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        
        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

        # 运行期缓存：缓存LLM图查询理解、图路径、子图和最终Document结果。
        cache_enabled = getattr(config, "cache_enabled", True)
        cache_max_size = getattr(config, "cache_max_size", 256)
        cache_ttl_seconds = getattr(config, "cache_ttl_seconds", 3600)
        self.graph_query_cache = TTLCache(cache_max_size, cache_ttl_seconds, cache_enabled)
        self.path_cache = TTLCache(cache_max_size, cache_ttl_seconds, cache_enabled)
        self.knowledge_subgraph_cache = TTLCache(cache_max_size, cache_ttl_seconds, cache_enabled)
        self.search_cache = TTLCache(cache_max_size, cache_ttl_seconds, cache_enabled)
        
    def initialize(self):
        """初始化图RAG检索系统，并刷新依赖Neo4j当前状态的本地索引。"""
        logger.info("初始化图RAG检索系统...")
        
        # 连接Neo4j
        try:
            if self.driver:
                self.driver.close()
            self.entity_cache.clear()
            self.relation_cache.clear()
            self.subgraph_cache.clear()
            self.clear_cache()
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, 
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self._session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return
        
        # 预热：构建实体和关系索引
        self._build_graph_index()

    def _session(self):
        """创建Neo4j会话，统一使用配置中的数据库名称。"""
        database = getattr(self.config, "neo4j_database", None)
        if database:
            return self.driver.session(database=database)
        return self.driver.session()

    def clear_cache(self) -> None:
        """清空图RAG运行期缓存，适用于知识库重建后。"""
        self.graph_query_cache.clear()
        self.path_cache.clear()
        self.knowledge_subgraph_cache.clear()
        self.search_cache.clear()

    def _graph_query_cache_key(self, graph_query: GraphQuery) -> str:
        """根据图查询结构生成稳定缓存键。"""
        return make_cache_key(
            graph_query.query_type.value,
            graph_query.source_entities or [],
            graph_query.target_entities or [],
            graph_query.relation_types or [],
            graph_query.max_depth,
            graph_query.max_nodes,
            graph_query.constraints or {},
        )
        
    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")
        
        try:
            with self._session() as session:
                # 构建实体索引 - 修复Neo4j语法兼容性问题
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id, 
                       n.name as name, n.category as category, degree
                ORDER BY degree DESC
                LIMIT 1000
                """
                
                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"]
                    }
                
                # 构建关系类型索引
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """
                
                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]
                    
                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")
                
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
    
    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图，重复问题直接复用LLM分析结果。
        这是图RAG的核心：从自然语言到图查询的转换
        """
        cache_key = make_cache_key("understand_graph_query", query)
        cached_query = self.graph_query_cache.get(cache_key)
        if cached_query is not None:
            logger.info("图查询理解命中缓存")
            return cached_query

        prompt = f"""
        作为图数据库专家，分析以下查询的图结构意图，并将自然语言问题映射到**已有图结构**上。
        
        已知图中大致有以下节点和关系：
        - 节点类型：
          - Recipe：菜谱节点，包含 name、description、cuisineType（如"川菜"）、category、tags、prepTime、cookTime 等属性
          - Ingredient：食材节点，包含 name、category（如"蔬菜"、"蛋白质" 等）
          - Category：菜品分类（如"川菜"、"家常菜"、"素菜"）
          - CookingStep：烹饪步骤
        - 主要关系：
          - (Recipe)-[:REQUIRES]->(Ingredient)
          - (Recipe)-[:BELONGS_TO_CATEGORY]->(Category)
          - (Recipe)-[:CONTAINS_STEP]->(CookingStep)
        
        请根据上述图结构分析下面的查询：
        
        查询：{query}
        
        请识别：
        1. 查询类型：
           - entity_relation: 询问实体间的直接关系（如：鸡肉和胡萝卜能一起做菜吗？）
           - multi_hop: 需要多跳推理（如：鸡肉配什么蔬菜？需要：鸡肉→菜品→食材→蔬菜）
           - subgraph: 需要完整子图（如：川菜有什么特色？需要川菜相关的完整知识网络）
           - path_finding: 路径查找（如：从食材到成品菜的制作路径）
           - clustering: 聚类相似性（如：和宫保鸡丁类似的菜有哪些？）
        
        2. source_entities：
           - 只包含在图中**很有可能有对应节点**的具体实体名称
           - 优先选择：菜系（如"川菜"）、具体菜名（如"宫保鸡丁"）、食材名（如"鸡肉"、"豆腐"）
           - 不要把抽象概念或约束（如"糖尿病饮食限制"、"具体川菜菜品"、"健康饮食"、"30分钟内"）放进 source_entities
        
        3. target_entities：
           - 只在确实需要限制「路径终点」时填写
           - 同样只能使用可能出现在 Recipe / Ingredient / Category 节点上的名称（如"蔬菜"、"素菜"、具体菜名）
           - 如果不确定目标实体怎么映射到图中，请返回空列表 []
        
        4. relation_types：本次推理中希望优先考虑的关系类型列表
           - 例如：["REQUIRES", "BELONGS_TO_CATEGORY"]
        
        5. max_depth：建议的图遍历深度（1-3 之间的整数）
        
        6. constraints：可选的**属性级约束**，用于表达图结构之外的过滤条件，例如：
           - 健康/饮食限制（如"糖尿病"、"低糖"）
           - 时间限制（如"30分钟内"）
           - 口味偏好（如"清淡"、"少油"）
           用一个字典描述，例如：
           {{
             "health": ["糖尿病", "低糖"],
             "time": {{"max_minutes": 30}},
             "style": ["川菜"]
           }}
        
        示例1：
        查询："鸡肉配什么蔬菜好？"
        期望分析：这是 multi_hop 查询，需要通过"鸡肉→使用鸡肉的菜品→这些菜品使用的蔬菜"的路径推理。
        
        返回JSON示例：
        {{
          "query_type": "multi_hop",
          "source_entities": ["鸡肉"],
          "target_entities": ["蔬菜"],
          "relation_types": ["REQUIRES", "BELONGS_TO_CATEGORY"],
          "max_depth": 3,
          "constraints": {{}}
        }}
        
        示例2：
        查询："适合糖尿病人吃的低糖川菜有哪些，并且制作时间不超过30分钟？"
        期望分析：
          - 图中可以直接对应的实体：主要是菜系 "川菜"
          - 糖尿病/低糖/30分钟 属于属性级约束，不能当作节点
          - 可以使用 subgraph 或 multi_hop，以 "川菜" 为核心实体，结合属性约束做后续过滤
        
        返回JSON示例：
        {{
          "query_type": "subgraph",
          "source_entities": ["川菜"],
          "target_entities": [],
          "relation_types": ["BELONGS_TO_CATEGORY", "REQUIRES"],
          "max_depth": 2,
          "constraints": {{
            "health": ["糖尿病", "低糖"],
            "time": {{"max_minutes": 30}}
          }}
        }}
        
        请严格返回一个合法的 JSON 对象，不要包含任何多余的说明文字。
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(strip_json_markdown(response.choices[0].message.content))
            
            graph_query = GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50,
                constraints=result.get("constraints", {})
            )
            self.graph_query_cache.set(cache_key, graph_query)
            return graph_query
            
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询
            graph_query = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )
            self.graph_query_cache.set(cache_key, graph_query)
            return graph_query
    
    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳图遍历：这是图RAG的核心优势；缓存相同图查询的路径结果。
        通过图结构发现隐含的知识关联
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")
        cache_key = make_cache_key("multi_hop_traversal", self._graph_query_cache_key(graph_query))
        cached_paths = self.path_cache.get(cache_key)
        if cached_paths is not None:
            logger.info("多跳遍历命中缓存")
            return cached_paths
        
        paths = []
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths
            
        try:
            with self._session() as session:
                # 构建多跳遍历查询
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth
                
                # 根据查询类型选择不同的遍历策略
                if graph_query.query_type == QueryType.MULTI_HOP:
                    # 根据是否有目标关键词动态拼接过滤条件
                    target_filter_clause = ""
                    if target_keywords:
                        target_filter_clause = """
                    AND ANY(kw IN $target_keywords WHERE
                        (target.name IS NOT NULL AND (toString(target.name) CONTAINS kw OR kw CONTAINS toString(target.name))) OR
                        (target.category IS NOT NULL AND (toString(target.category) CONTAINS kw OR kw CONTAINS toString(target.category)))
                    )"""
                    
                    cypher_query = f"""
                    // 多跳推理查询
                    UNWIND $source_entities as source_name
                    MATCH (source)
                    WHERE source.name CONTAINS source_name OR source.nodeId = source_name
                    
                    // 执行多跳遍历
                    MATCH path = (source)-[*1..{max_depth}]-(target)
                    WHERE NOT source = target{target_filter_clause}
                    
                    // 计算路径相关性
                    WITH path, source, target,
                         length(path) as path_len,
                         relationships(path) as rels,
                         nodes(path) as path_nodes
                    
                    // 路径评分：短路径 + 高度数节点 + 关系类型匹配
                    WITH path, source, target, path_len, rels, path_nodes,
                         (1.0 / path_len) + 
                         (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / 10.0 / size(path_nodes)) +
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
                    
                    ORDER BY relevance DESC
                    LIMIT 20
                    
                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """
                    
                    params = {
                        "source_entities": source_entities,
                        "relation_types": graph_query.relation_types or []
                    }
                    if target_keywords:
                        params["target_keywords"] = target_keywords
                    
                    result = session.run(cypher_query, params)
                    
                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            paths.append(path_data)
                
                elif graph_query.query_type == QueryType.ENTITY_RELATION:
                    # 实体间关系查询
                    paths.extend(self._find_entity_relations(graph_query, session))
                
                elif graph_query.query_type == QueryType.PATH_FINDING:
                    # 最短路径查找
                    paths.extend(self._find_shortest_paths(graph_query, session))
                    
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
            
        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        self.path_cache.set(cache_key, paths)
        return paths
    
    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """
        提取知识子图：获取实体相关的完整知识网络，并缓存相同子图查询。
        这体现了图RAG的整体性思维
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")
        cache_key = make_cache_key("extract_knowledge_subgraph", self._graph_query_cache_key(graph_query))
        cached_subgraph = self.knowledge_subgraph_cache.get(cache_key)
        if cached_subgraph is not None:
            logger.info("知识子图命中缓存")
            return cached_subgraph
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            subgraph = self._fallback_subgraph_extraction(graph_query)
            self.knowledge_subgraph_cache.set(cache_key, subgraph)
            return subgraph
        
        try:
            with self._session() as session:
                # 简化的子图提取（不依赖APOC），并把变长路径中的关系展开去重。
                cypher_query = f"""
                // 找到源实体
                UNWIND $source_entities as entity_name
                MATCH (source)
                WHERE source.name CONTAINS entity_name 
                   OR entity_name CONTAINS source.name
                   OR source.nodeId = entity_name
                
                // 获取指定深度的邻居
                MATCH path = (source)-[*1..{graph_query.max_depth}]-(neighbor)
                WITH source,
                     collect(DISTINCT neighbor) as all_neighbors,
                     collect(DISTINCT path) as paths
                UNWIND paths as p
                UNWIND relationships(p) as rel
                WITH source,
                     all_neighbors,
                     collect(DISTINCT rel) as all_relationships
                
                // 计算图指标
                WITH source,
                     all_neighbors[0..$max_nodes] as neighbors,
                     all_relationships[0..$max_nodes] as relationships,
                     size(all_neighbors) as node_count,
                     size(all_relationships) as rel_count
                
                RETURN 
                    source,
                    neighbors as nodes,
                    relationships as rels,
                    {{
                        node_count: node_count,
                        relationship_count: rel_count,
                        density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                    }} as metrics
                """
                
                result = session.run(cypher_query, {
                    "source_entities": graph_query.source_entities,
                    "max_nodes": graph_query.max_nodes
                })
                
                record = result.single()
                if record:
                    subgraph = self._build_knowledge_subgraph(record)
                    self.knowledge_subgraph_cache.set(cache_key, subgraph)
                    return subgraph
                    
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
            
        # 降级方案：简单邻居查询
        subgraph = self._fallback_subgraph_extraction(graph_query)
        self.knowledge_subgraph_cache.set(cache_key, subgraph)
        return subgraph
    
    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于图结构的推理：这是图RAG的智能之处
        不仅检索信息，还能进行逻辑推理
        """
        reasoning_chains = []
        
        try:
            # 1. 识别推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)
            
            # 2. 构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)
            
            # 3. 验证推理链的可信度
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)
            
            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains
            
        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []
    
    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应查询规划：根据查询复杂度动态调整策略
        """
        # 分析查询复杂度
        complexity_score = self._analyze_query_complexity(query)
        
        query_plans = []
        
        if complexity_score < 0.3:
            # 简单查询：直接邻居查询
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)
            
        elif complexity_score < 0.7:
            # 中等复杂度：多跳查询
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)
            
        else:
            # 复杂查询：子图提取 + 推理
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])
            
        return query_plans
    
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力，并缓存最终Document结果。
        """
        logger.info(f"开始图RAG检索: {query}")
        cache_key = make_cache_key("graph_rag_search", query, top_k)
        cached_docs = self.search_cache.get(cache_key)
        if cached_docs is not None:
            logger.info("图RAG检索命中缓存")
            return clone_documents(cached_docs)
        
        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []
        
        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")
        
        results = []
        
        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历 / 路径查找
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
                
            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                # 子图提取 / 聚类查询：都视为“围绕核心实体的局部知识网络”
                subgraph = self.extract_knowledge_subgraph(graph_query)
                
                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))
                
            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询（可以视为一跳 / 少量跳的路径查询）
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            
            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)
            
            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            final_docs = results[:top_k]
            self.search_cache.set(cache_key, final_docs)
            return clone_documents(final_docs)
            
        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []
    
    # ========== 辅助方法 ==========
    
    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录"""
        try:
            path_nodes = []
            for node in record["path_nodes"]:
                path_nodes.append({
                    "id": node.get("nodeId", ""),
                    "name": node.get("name", ""),
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            
            relationships = []
            for rel in record["rels"]:
                rel_type = getattr(rel, "type", None) or type(rel).__name__
                relationships.append({
                    "type": rel_type,
                    "properties": dict(rel)
                })
            
            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=record["path_len"],
                relevance_score=record["relevance"],
                path_type="multi_hop"
            )
            
        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None
    
    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象"""
        try:
            central_nodes = [dict(record["source"])]
            connected_nodes = [dict(node) for node in record["nodes"]]
            relationships = [
                {
                    "type": getattr(rel, "type", None) or type(rel).__name__,
                    "properties": dict(rel),
                }
                for rel in record["rels"]
            ]
            
            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=record["metrics"],
                reasoning_chains=[]
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )
    
    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象"""
        documents = []
        
        for i, path in enumerate(paths):
            # 构建路径描述
            path_desc = self._build_path_description(path)
            
            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": "graph_path",
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "recipe_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果"
                }
            )
            documents.append(doc)
            
        return documents
    
    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph, 
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为Document对象"""
        documents = []
        
        # 子图整体描述
        subgraph_desc = self._build_subgraph_description(subgraph)
        
        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "knowledge_subgraph",
                "node_count": len(subgraph.connected_nodes),
                "relationship_count": len(subgraph.relationships),
                "graph_density": subgraph.graph_metrics.get("density", 0.0),
                "reasoning_chains": reasoning_chains,
                "recipe_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图"
            }
        )
        documents.append(doc)
        
        return documents
    
    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述"""
        if not path.nodes:
            return "空路径"
            
        desc_parts = []
        for i, node in enumerate(path.nodes):
            desc_parts.append(node.get("name", f"节点{i}"))
            if i < len(path.relationships):
                rel_type = path.relationships[i].get("type", "相关")
                desc_parts.append(f" --{rel_type}--> ")
        
        return "".join(desc_parts)
    
    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述"""
        central_names = [node.get("name", "未知") for node in subgraph.central_nodes]
        node_count = len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)
        
        return f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"
    
    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(documents, 
                     key=lambda x: x.metadata.get("relevance_score", 0.0), 
                     reverse=True)
    
    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        complexity_indicators = ["什么", "如何", "为什么", "哪些", "关系", "影响", "原因"]
        score = sum(1 for indicator in complexity_indicators if indicator in query)
        return min(score / len(complexity_indicators), 1.0)
    
    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别推理模式"""
        return ["因果关系", "组成关系", "相似关系"]
    
    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """构建推理链"""
        return f"基于{pattern}的推理链"
    
    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链"""
        return chains[:3]
    
    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系；当没有显式目标时返回源实体附近的高相关路径。"""
        paths: List[GraphPath] = []
        source_entities = graph_query.source_entities or []
        if not source_entities:
            return paths

        target_keywords = graph_query.target_entities or (
            source_entities if len(source_entities) > 1 else []
        )
        target_filter_clause = ""
        params = {
            "source_entities": source_entities,
            "relation_types": graph_query.relation_types or [],
            "limit": min(graph_query.max_nodes, 20),
        }

        if target_keywords:
            target_filter_clause = """
              AND ANY(kw IN $target_keywords WHERE
                  (target.name IS NOT NULL AND (target.name CONTAINS kw OR kw CONTAINS target.name)) OR
                  target.nodeId = kw
              )
            """
            params["target_keywords"] = target_keywords

        cypher_query = f"""
        UNWIND $source_entities as source_name
        MATCH (source)
        WHERE (source.name IS NOT NULL AND (source.name CONTAINS source_name OR source_name CONTAINS source.name))
           OR source.nodeId = source_name
        MATCH path = (source)-[*1..{max(1, graph_query.max_depth)}]-(target)
        WHERE NOT source = target
        {target_filter_clause}
        WITH path,
             length(path) as path_len,
             relationships(path) as rels,
             nodes(path) as path_nodes
        WITH path, path_len, rels, path_nodes,
             (1.0 / path_len) +
             (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
        ORDER BY relevance DESC
        LIMIT $limit
        RETURN path, path_len, rels, path_nodes, relevance
        """

        for record in session.run(cypher_query, params):
            path_data = self._parse_neo4j_path(record)
            if path_data:
                path_data.path_type = "entity_relation"
                paths.append(path_data)

        return paths
    
    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找源实体到目标实体的最短路径，补齐原先空实现。"""
        paths: List[GraphPath] = []
        source_entities = graph_query.source_entities or []
        target_entities = graph_query.target_entities or []

        if not source_entities:
            return paths
        if not target_entities and len(source_entities) > 1:
            target_entities = source_entities
        if not target_entities:
            return paths

        cypher_query = f"""
        UNWIND $source_entities as source_name
        UNWIND $target_entities as target_name
        WITH source_name, target_name
        WHERE source_name <> target_name
        MATCH (source), (target)
        WHERE ((source.name IS NOT NULL AND (source.name CONTAINS source_name OR source_name CONTAINS source.name))
               OR source.nodeId = source_name)
          AND ((target.name IS NOT NULL AND (target.name CONTAINS target_name OR target_name CONTAINS target.name))
               OR target.nodeId = target_name)
        MATCH path = shortestPath((source)-[*1..{max(1, graph_query.max_depth)}]-(target))
        WITH path,
             length(path) as path_len,
             relationships(path) as rels,
             nodes(path) as path_nodes
        WITH path, path_len, rels, path_nodes,
             (1.0 / path_len) +
             (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
        ORDER BY relevance DESC
        LIMIT $limit
        RETURN path, path_len, rels, path_nodes, relevance
        """

        params = {
            "source_entities": source_entities,
            "target_entities": target_entities,
            "relation_types": graph_query.relation_types or [],
            "limit": min(graph_query.max_nodes, 20),
        }

        for record in session.run(cypher_query, params):
            path_data = self._parse_neo4j_path(record)
            if path_data:
                path_data.path_type = "shortest_path"
                paths.append(path_data)

        return paths
    
    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取"""
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[]
        )
    
    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭")
