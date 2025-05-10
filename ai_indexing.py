import logging
from typing import List, Optional, Dict, Any
import numpy as np
import faiss
import sqlite3

logger = logging.getLogger(__name__)

class AIIndexing:
    def __init__(self, embedding_store, tabular_store):
        self.embedding_store = embedding_store
        self.tabular_store = tabular_store

    def analyze_embedding_data(self):
        """
        Analyze embedding data characteristics to recommend index type.
        """
        num_vectors = self.embedding_store.index.ntotal
        dim = self.embedding_store.dimension
        logger.info(f"Analyzing embedding data: {num_vectors} vectors of dimension {dim}")

        # Simple heuristic: if data is large, use IVF index, else flat index
        if num_vectors > 10000:
            recommended_index = "IVF"
        else:
            recommended_index = "Flat"
        logger.info(f"Recommended embedding index type: {recommended_index}")
        return recommended_index

    def build_embedding_index(self, index_type: Optional[str] = None):
        """
        Build or rebuild the embedding index based on recommended or specified type.
        """
        if index_type is None:
            index_type = self.analyze_embedding_data()

        dim = self.embedding_store.dimension
        if index_type == "IVF":
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            # Train index with existing vectors
            vectors = self.embedding_store.get_all_vectors()
            if vectors.shape[0] == 0:
                logger.warning("No vectors to train IVF index.")
                return
            index.train(vectors)
            index.add(vectors)
            self.embedding_store.index = index
            logger.info("Built IVF index for embeddings.")
        else:
            # Flat index
            index = faiss.IndexFlatL2(dim)
            vectors = self.embedding_store.get_all_vectors()
            if vectors.shape[0] > 0:
                index.add(vectors)
            self.embedding_store.index = index
            logger.info("Built Flat index for embeddings.")

    def analyze_tabular_queries(self, query_samples: List[str]):
        """
        Analyze tabular query samples to recommend SQLite indexes.
        """
        # For simplicity, parse queries to find columns used in WHERE clauses
        columns = set()
        for query in query_samples:
            # Very naive parsing: look for "WHERE <column> ..."
            tokens = query.lower().split()
            if "where" in tokens:
                where_index = tokens.index("where")
                if where_index + 1 < len(tokens):
                    col = tokens[where_index + 1]
                    columns.add(col)
        logger.info(f"Recommended SQLite indexes on columns: {columns}")
        return columns

    def create_sqlite_indexes(self, columns: List[str]):
        """
        Create SQLite indexes on specified columns.
        """
        conn = self.tabular_store.conn
        cursor = conn.cursor()
        for col in columns:
            try:
                index_name = f"idx_{col}"
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON records(json_extract(fields, '$.{col}'))")
                logger.info(f"Created index {index_name} on column {col}")
            except Exception as e:
                logger.error(f"Failed to create index on {col}: {e}")
        conn.commit()

    def optimize(self, query_samples: Optional[List[str]] = None):
        """
        Run the full auto-indexing and optimization process.
        """
        self.build_embedding_index()
        if query_samples:
            columns = self.analyze_tabular_queries(query_samples)
            if columns:
                self.create_sqlite_indexes(list(columns))
        logger.info("Auto-indexing and optimization complete.")
