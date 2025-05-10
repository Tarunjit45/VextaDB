from typing import List, Optional
import numpy as np
from core.models import EmbeddingRecord, TabularRecord
from vector_search.embedding_store import EmbeddingStore
from relational.tabular_store import TabularStore
from ai_indexing import AIIndexing
from self_optimization import SelfOptimizationEngine

class AIDatabase:
    def __init__(self, embedding_dim: int, tabular_db_path: str = ":memory:", privacy_preserving: bool = False, epsilon: float = 1.0):
        self.embedding_store = EmbeddingStore(embedding_dim, privacy_preserving=privacy_preserving, epsilon=epsilon)
        self.tabular_store = TabularStore(tabular_db_path)
        self.auto_indexer = AIIndexing(self.embedding_store, self.tabular_store)
        self.self_optimizer = SelfOptimizationEngine(self)

    def insert_embedding(self, id: str, embedding: np.ndarray, metadata: Optional[dict] = None):
        record = EmbeddingRecord(id=id, embedding=embedding, metadata=metadata)
        self.embedding_store.add(record)

    def query_embedding(self, query_vector: np.ndarray, top_k: int = 5) -> List[EmbeddingRecord]:
        return self.embedding_store.search(query_vector, top_k)

    def insert_tabular(self, id: str, fields: dict):
        record = TabularRecord(id=id, fields=fields)
        self.tabular_store.insert(record)

    def query_tabular(self, filter_func: Optional[callable] = None) -> List[TabularRecord]:
        return self.tabular_store.query(filter_func)

    def run_auto_indexing(self, query_samples: Optional[List[str]] = None):
        """
        Run AI-based auto-indexing and optimization.
        """
        self.auto_indexer.optimize(query_samples)

    def start_self_optimization(self):
        """
        Start the self-optimization engine.
        """
        self.self_optimizer.start()

    def stop_self_optimization(self):
        """
        Stop the self-optimization engine.
        """
        self.self_optimizer.stop()
