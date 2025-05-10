import numpy as np
import sys
import os

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.python_api import AIDatabase

def main():
    db = AIDatabase(128)

    # Example: Query embeddings similar to a random vector
    query_vector = np.random.rand(128).astype('float32')
    embedding_results = db.query_embedding(query_vector, top_k=5)
    print("Embedding search results:")
    for record in embedding_results:
        print(f"ID: {record.id} Metadata: {record.metadata}")

    # Example: Query tabular data for users with last_login before 2024-12-15
    def filter_func(record):
        last_login = record.fields.get('last_login', '')
        return last_login < '2024-12-15'

    tabular_results = db.query_tabular(filter_func)
    print("Tabular query results:")
    for record in tabular_results:
        print(f"ID: {record.id} Fields: {record.fields}")

if __name__ == "__main__":
    main()
