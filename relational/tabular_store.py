import sqlite3
from typing import Dict, Any, List, Optional
from core.models import TabularRecord

class TabularStore:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_table()

    def _initialize_table(self):
        # For simplicity, create a generic table with id and JSON fields
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            fields TEXT
        )
        """)
        self.conn.commit()

    def insert(self, record: TabularRecord):
        import json
        fields_json = json.dumps(record.fields)
        self.cursor.execute("""
        INSERT OR REPLACE INTO records (id, fields) VALUES (?, ?)
        """, (record.id, fields_json))
        self.conn.commit()

    def query(self, filter_func: Optional[Any] = None) -> List[TabularRecord]:
        # For simplicity, fetch all and filter in Python
        self.cursor.execute("SELECT id, fields FROM records")
        rows = self.cursor.fetchall()
        import json
        results = []
        for row in rows:
            id_, fields_json = row
            fields = json.loads(fields_json)
            record = TabularRecord(id=id_, fields=fields)
            if filter_func is None or filter_func(record):
                results.append(record)
        return results
