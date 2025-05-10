import faiss
import numpy as np
from typing import List, Optional, Dict
from core.models import EmbeddingRecord

class EmbeddingStore:
    def __init__(self, dimension: int, privacy_preserving: bool = False, epsilon: float = 1.0):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.id_to_metadata: Dict[int, EmbeddingRecord] = {}
        self.next_id = 0
        self.vectors = []
        self.privacy_preserving = privacy_preserving
        self.epsilon = epsilon  # Privacy budget for differential privacy

    def add(self, embedding_record: EmbeddingRecord):
        vector = np.array(embedding_record.embedding).astype('float32').reshape(1, -1)
        if self.privacy_preserving:
            vector = self._add_dp_noise(vector)
        self.index.add(vector)
        self.id_to_metadata[self.next_id] = embedding_record
        self.next_id += 1
        self.vectors.append(vector)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[EmbeddingRecord]:
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        if self.privacy_preserving:
            query_vector = self._add_dp_noise(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.id_to_metadata.get(idx))
        return results

    def get_all_vectors(self) -> np.ndarray:
        if not self.vectors:
            return np.empty((0, self.dimension), dtype='float32')
        return np.vstack(self.vectors)

    def _add_dp_noise(self, vector: np.ndarray) -> np.ndarray:
        # Add Laplace noise for differential privacy
        sensitivity = 1.0  # Sensitivity of the embedding vector (assumed)
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, vector.shape).astype('float32')
        noisy_vector = vector + noise
        return noisy_vector
