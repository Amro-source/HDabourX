import sqlite3

conn = sqlite3.connect("notes.db")
cursor = conn.cursor()
cursor.execute("SELECT id, name, parent_id FROM Tags")
tags = cursor.fetchall()
conn.close()

print(tags)  # ✅ Check if any tags exist
