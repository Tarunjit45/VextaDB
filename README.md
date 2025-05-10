# AI-Native Hybrid Vector + Relational Database

## Overview

This project is a custom AI-native database built in Python that combines hybrid vector and relational storage with advanced AI-driven features. It supports storing and querying both high-dimensional embeddings (for similarity search) and structured tabular data (SQL-like queries).

### Key Features

1. **Hybrid Vector + Relational Storage**
   - Store and retrieve high-dimensional embeddings using FAISS.
   - Store and query structured tabular data using SQLite.
   - Support combined queries filtering by vector similarity and tabular conditions.

2. **Natural Language Query Interface**
   - (Planned) Use large language models (LLMs) to parse natural language queries into database actions.

3. **AI-based Auto-Indexing**
   - Monitor query patterns and automatically suggest or create optimal indexes to improve performance.

4. **Self-Optimization Engine**
   - Reinforcement learning agent that continuously monitors query performance and dynamically adjusts indexing strategies.

5. **Privacy-Preserving Search**
   - Differential privacy support by adding calibrated noise to embeddings and queries to protect sensitive data.

6. **Simple Developer API**
   - Python API for inserting embeddings and tabular records, running queries, and controlling optimization features.

## Getting Started

### Installation

Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

```python
from api.python_api import AIDatabase
import numpy as np

# Initialize database with 128-dimensional embeddings and privacy-preserving enabled
db = AIDatabase(embedding_dim=128, privacy_preserving=True, epsilon=1.0)

# Insert embedding
embedding_vector = np.random.rand(128).astype('float32')
db.insert_embedding(id="user1", embedding=embedding_vector, metadata={"name": "Alice"})

# Insert tabular data
db.insert_tabular(id="user1", fields={"last_login": "2024-12-01", "age": 30})

# Query embeddings
query_vector = np.random.rand(128).astype('float32')
results = db.query_embedding(query_vector, top_k=5)

# Query tabular data
results = db.query_tabular(filter_func=lambda record: record.fields.get("age", 0) > 25)

# Run AI-based auto-indexing
db.run_auto_indexing(query_samples=["SELECT * FROM users WHERE last_login < '2025-01-01'"])

# Start self-optimization engine
db.start_self_optimization()

# Stop self-optimization engine
db.stop_self_optimization()
```

## Project Structure

- `/core`: Core data models and storage abstractions
- `/vector_search`: Embedding storage and similarity search (FAISS)
- `/relational`: Tabular data storage and SQL-like query support (SQLite)
- `/nlp`: Natural language query parsing and mapping to database actions
- `/api`: REST and Python API implementations
- `/optimization`: AI-based auto-indexing and self-optimization engine
- `/privacy`: Privacy-preserving search features
- `/demos`: Sample demo use cases
- `/tests`: Unit and integration tests
- `/scripts`: Utility scripts for setup and maintenance

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements or new features.

## License

This project is licensed under the MIT License.
