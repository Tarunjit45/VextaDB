class NLQueryParser:
    """
    Mock Natural Language Query Parser for testing without loading large models.
    """

    def __init__(self, *args, **kwargs):
        pass

    def parse(self, query: str) -> dict:
        # Return a fixed command for testing
        if "similar to this image" in query.lower():
            return {
                "action": "search_embeddings",
                "parameters": {
                    "query_vector": None,
                    "top_k": 3
                }
            }
        elif "last logged in before" in query.lower():
            # Example filter function for tabular query
            def filter_func(record):
                last_login = record.fields.get("last_login", "")
                return last_login < "2025-01-01"
            return {
                "action": "query_tabular",
                "parameters": {
                    "filter_func": filter_func
                }
            }
        else:
            return {
                "action": "search_embeddings",
                "parameters": {
                    "query_vector": None,
                    "top_k": 5
                }
            }
