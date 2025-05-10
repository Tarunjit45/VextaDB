from nlp.nl_query_parser import NLQueryParser
from api.python_api import AIDatabase
import numpy as np

class NLQueryInterface:
    def __init__(self, embedding_dim: int):
        self.db = AIDatabase(embedding_dim)
        self.parser = NLQueryParser()

    def query(self, natural_language_query: str):
        # Parse the natural language query to get structured command
        command = self.parser.parse(natural_language_query)

        # Example: command could be a dict with action and parameters
        action = command.get("action")
        params = command.get("parameters", {})

        if action == "search_embeddings":
            query_vector = params.get("query_vector")
            top_k = params.get("top_k", 5)
            if query_vector is None:
                # For demo, generate a random vector (replace with real embedding extraction)
                query_vector = np.random.rand(self.db.embedding_store.dimension).astype('float32')
            results = self.db.query_embedding(query_vector, top_k)
            return results

        elif action == "query_tabular":
            filter_func = params.get("filter_func")
            results = self.db.query_tabular(filter_func)
            return results

        else:
            raise ValueError(f"Unknown action: {action}")
