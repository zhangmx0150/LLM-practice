"""
轻量缓存与序列化工具。

这些工具只缓存纯数据结果，不改变检索、排序和生成的核心逻辑。
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document


class TTLCache:
    """线程安全的 TTL + LRU 内存缓存。"""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600, enabled: bool = True):
        """初始化缓存容量、过期时间和开关。"""
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.enabled = enabled
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: str, default: Any = None) -> Any:
        """读取缓存值；命中时返回深拷贝，避免调用方修改缓存内对象。"""
        if not self.enabled:
            return default

        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return default

            created_at, value = item
            if now - created_at > self.ttl_seconds:
                self._store.pop(key, None)
                return default

            self._store.move_to_end(key)
            return copy.deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        """写入缓存值；写入时深拷贝，隔离后续外部修改。"""
        if not self.enabled:
            return

        with self._lock:
            self._store[key] = (time.monotonic(), copy.deepcopy(value))
            self._store.move_to_end(key)
            self._evict_if_needed()

    def clear(self) -> None:
        """清空全部缓存内容。"""
        with self._lock:
            self._store.clear()

    def _evict_if_needed(self) -> None:
        """超过容量时按最久未使用顺序淘汰旧条目。"""
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)


def stable_json_dumps(value: Any) -> str:
    """把常见 Python 对象稳定序列化，便于构建缓存键。"""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def make_cache_key(*parts: Any) -> str:
    """根据任意输入片段生成短且稳定的 SHA256 缓存键。"""
    raw = stable_json_dumps(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def strip_json_markdown(content: str) -> str:
    """清理大模型常见的 ```json 代码块包裹，返回可解析的 JSON 字符串。"""
    cleaned = (content or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def clone_document(document: Document) -> Document:
    """复制 Document，避免缓存命中对象被调用方原地修改。"""
    return Document(
        page_content=document.page_content,
        metadata=dict(document.metadata),
    )


def clone_documents(documents: Iterable[Document]) -> List[Document]:
    """批量复制 Document 列表。"""
    return [clone_document(doc) for doc in documents]


def document_fingerprint(document: Document) -> Dict[str, Any]:
    """生成文档指纹，用于回答缓存判断上下文是否相同。"""
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    metadata = document.metadata or {}
    return {
        "content_hash": content_hash,
        "node_id": metadata.get("node_id"),
        "chunk_id": metadata.get("chunk_id"),
        "recipe_name": metadata.get("recipe_name"),
        "search_type": metadata.get("search_type"),
        "final_score": metadata.get("final_score"),
        "relevance_score": metadata.get("relevance_score"),
    }
