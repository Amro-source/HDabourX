import sqlite3

conn = sqlite3.connect("notes.db")
cursor = conn.cursor()

# Insert Sample Tags (If not already present)
cursor.executemany("INSERT OR IGNORE INTO Tags (id, name, parent_id) VALUES (?, ?, ?)", [
    (1, "Work", None),
    (2, "Personal", None),
    (3, "Projects", 1),  # Child of "Work"
    (4, "Ideas", 1),  # Child of "Work"
    (5, "Tasks", 2)  # Child of "Personal"
])

conn.commit()
conn.close()

print("Sample tags added successfully!")
