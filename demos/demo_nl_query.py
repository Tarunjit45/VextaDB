import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from nlp.nl_query_interface import NLQueryInterface

def main():
    embedding_dim = 128
    nl_interface = NLQueryInterface(embedding_dim)

    # Example natural language queries
    queries = [
        "Find users similar to this image of a face.",
        "Show all users who last logged in before 2025-01-01."
    ]

    for q in queries:
        print(f"Query: {q}")
        try:
            results = nl_interface.query(q)
            for r in results:
                if hasattr(r, 'id'):
                    print(f"Result ID: {r.id}")
                else:
                    print(r)
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
