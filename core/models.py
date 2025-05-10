from typing import Any, Dict, List, Optional
import numpy as np

class EmbeddingRecord:
    """
    Data model for an embedding record.
    """
    def __init__(self, id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.embedding = embedding
        self.metadata = metadata or {}

class TabularRecord:
    """
    Data model for a tabular record.
    """
    def __init__(self, id: str, fields: Dict[str, Any]):
        self.id = id
        self.fields = fields
