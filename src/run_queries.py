
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "instamart.db"
SQL_PATH = BASE_DIR / "sql" / "queries.sql"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

sql_text = SQL_PATH.read_text(encoding="utf-8")
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

with sqlite3.connect(DB_PATH) as connection:
    for i, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, connection)

        output_file = OUTPUT_DIR / f"query_{i}_results.csv"
        result.to_csv(output_file, index=False)

        print(f"Query {i} saved: {output_file.name}")

print("\nALL SEVEN QUERIES SAVED SUCCESSFULLY")
