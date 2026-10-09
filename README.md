# ⚡ VextaDB — Autonomous Self-Optimizing Hybrid Vector & Relational Database

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FAISS](https://img.shields.io/badge/Vector-FAISS-00599C?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![SQLite](https://img.shields.io/badge/Relational-SQLite3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org/)
[![Reinforcement Learning](https://img.shields.io/badge/Engine-Self--Optimizing%20RL-00C853?style=for-the-badge)](self_optimization.py)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**VextaDB** is an autonomous hybrid database engine that natively unifies **high-dimensional vector similarity search** with **structured relational data storage**. Equipped with an integrated **Reinforcement Learning optimization agent** (`self_optimization.py`), VextaDB dynamically self-tunes its indexing topology (FAISS index types, cluster centroids, quantization parameters) based on continuous runtime query patterns.

---

## 🧠 Architectural Overview

```
                          [ Incoming Query Stream ]
                          (SQL Relational + Vector)
                                      |
                                      v
+---------------------------------------------------------------------------+
|                          VextaDB Hybrid Engine                            |
+---------------------------------------------------------------------------+
               |                                            |
               v                                            v
+-----------------------------+              +-----------------------------+
|    Relational Subsystem     |              |     Vector Subsystem        |
|  (`relational/`, SQLite3)   |              |  (`vector_search/`, FAISS)  |
|  Tabular schema, ACID joins |              |  Dense embedding similarity |
+-----------------------------+              +-----------------------------+
               |                                            |
               +----------------------+---------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
|               Reinforcement Learning Optimizer (`self_optimization.py`)   |
|   Q-Learning Agent monitors query latency, hit rate & memory footprint    |
|   Dynamically re-indexes (IVF-PQ vs. Flat) to optimize Pareto frontier    |
+---------------------------------------------------------------------------+
```

---

## ✨ Features

* 🔀 **Hybrid Query Engine:** Execute joint queries filtering by structured metadata (e.g. `price < 100`, `category = 'electronics'`) while simultaneously computing vector cosine distance.
* 🤖 **Autonomous Index Tuning:** The RL agent (`ReinforcementLearningAgent`) observes latency rewards and automatically selects optimal indexing strategies.
* ⚡ **High-Throughput Vector Indexing:** Hardware-accelerated FAISS integration supporting millions of vectors.
* 📊 **Integrated API & Demos:** Ready-to-run interactive demonstrations in `demos/`.

---

## 📁 Repository Structure

```text
VextaDB/
├── ai_indexing.py          # AI-driven embedding & tabular analysis coordinator
├── self_optimization.py    # Reinforcement Learning agent for automatic index tuning
├── vector_search/          # FAISS vector similarity search modules
├── relational/             # SQLite tabular storage engine & SQL parsers
├── core/                   # Storage manager, cache buffer & transaction log
├── api/                    # REST / RPC API endpoints
├── demos/                  # Interactive benchmarks & demonstration scripts
├── nlp/                    # Text-to-embedding translation pipeline
├── LICENSE                 # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Tarunjit45/VextaDB.git
cd VextaDB

pip install faiss-cpu numpy sqlite3
```

### 2. Run Interactive Demo
```bash
python demos/basic_search.py
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
