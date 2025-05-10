import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from api.python_api import AIDatabase

def main():
    # Initialize database with embedding dimension 128
    db = AIDatabase(embedding_dim=128)

    # Insert some embeddings
    embedding1 = np.random.rand(128).astype('float32')
    embedding2 = np.random.rand(128).astype('float32')
    db.insert_embedding("emb1", embedding1, metadata={"type": "user"})
    db.insert_embedding("emb2", embedding2, metadata={"type": "user"})

    # Query embeddings similar to embedding1
    results = db.query_embedding(embedding1, top_k=2)
    print("Embedding search results:")
    for r in results:
        print(f"ID: {r.id}, Metadata: {r.metadata}")

    # Insert tabular data
    db.insert_tabular("user1", {"name": "Alice", "last_login": "2024-12-01"})
    db.insert_tabular("user2", {"name": "Bob", "last_login": "2025-02-15"})

    # Query tabular data with filter
    def filter_func(record):
        return record.fields.get("last_login", "") < "2025-01-01"

    tabular_results = db.query_tabular(filter_func)
    print("Tabular query results:")
    for r in tabular_results:
        print(f"ID: {r.id}, Fields: {r.fields}")

if __name__ == "__main__":
    main()
