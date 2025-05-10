import numpy as np
import random
import string
import sys
import os

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.python_api import AIDatabase

def random_id(n=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def main():
    db = AIDatabase(128)
    for i in range(50):
        embedding = np.random.rand(128).astype('float32')
        db.insert_embedding(random_id(), embedding, metadata={'type': 'user', 'index': i})
        db.insert_tabular(random_id(), {'name': f'User{i}', 'last_login': f'2024-12-{(i%30)+1:02d}'})
        print(f'Inserted record {i+1}')

if __name__ == "__main__":
    main()
