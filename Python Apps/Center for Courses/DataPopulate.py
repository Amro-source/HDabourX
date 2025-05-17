import sqlite3

def initialize_database():
    """Initialize the database structure"""
    conn = sqlite3.connect('course_center.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        registration_date TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS teachers (
        teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        specialization TEXT,
        hire_date TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        course_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        start_date TEXT,
        end_date TEXT,
        capacity INTEGER,
        price REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS student_courses (
        student_id INTEGER,
        course_id INTEGER,
        enrollment_date TEXT DEFAULT CURRENT_TIMESTAMP,
        grade REAL,
        PRIMARY KEY (student_id, course_id),
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS teacher_courses (
        teacher_id INTEGER,
        course_id INTEGER,
        assignment_date TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (teacher_id, course_id),
        FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    )
    ''')
    
    conn.commit()
    conn.close()

class CourseCenterApp:
    def __init__(self):
        self.conn = sqlite3.connect('course_center.db')
        self.cursor = self.conn.cursor()
    
    def add_student(self, first_name, last_name, email, phone):
        try:
            self.cursor.execute('''
            INSERT INTO students (first_name, last_name, email, phone)
            VALUES (?, ?, ?, ?)
            ''', (first_name, last_name, email, phone))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def add_teacher(self, first_name, last_name, email, phone, specialization):
        try:
            self.cursor.execute('''
            INSERT INTO teachers (first_name, last_name, email, phone, specialization)
            VALUES (?, ?, ?, ?, ?)
            ''', (first_name, last_name, email, phone, specialization))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def add_course(self, title, description, start_date, end_date, capacity, price):
        try:
            self.cursor.execute('''
            INSERT INTO courses (title, description, start_date, end_date, capacity, price)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (title, description, start_date, end_date, capacity, price))
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False
    
    def enroll_student(self, student_id, course_id):
        try:
            self.cursor.execute('''
            INSERT INTO student_courses (student_id, course_id)
            VALUES (?, ?)
            ''', (student_id, course_id))
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False
    
    def assign_teacher(self, teacher_id, course_id):
        try:
            self.cursor.execute('''
            INSERT INTO teacher_courses (teacher_id, course_id)
            VALUES (?, ?)
            ''', (teacher_id, course_id))
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False
    
    def list_students(self):
        self.cursor.execute('SELECT * FROM students')
        return self.cursor.fetchall()
    
    def list_courses(self):
        self.cursor.execute('SELECT * FROM courses')
        return self.cursor.fetchall()

def populate_sample_data():
    """Populate the database with sample data"""
    initialize_database()
    app = CourseCenterApp()
    
    # Add sample students
    students = [
        ("John", "Doe", "john@example.com", "1234567890"),
        ("Jane", "Smith", "jane@example.com", "0987654321"),
        ("Mike", "Brown", "mike@example.com", "5551234567"),
        ("Sarah", "Johnson", "sarah@example.com", "5559876543")
    ]
    for student in students:
        app.add_student(*student)
    
    # Add sample teachers
    teachers = [
        ("Alice", "Williams", "alice@example.com", "1112223333", "Mathematics"),
        ("Bob", "Miller", "bob@example.com", "4445556666", "Physics"),
        ("Carol", "Davis", "carol@example.com", "7778889999", "Computer Science")
    ]
    for teacher in teachers:
        app.add_teacher(*teacher)
    
    # Add sample courses
    courses = [
        ("Math 101", "Introduction to Algebra", "2023-09-01", "2023-12-15", 30, 299.99),
        ("Physics 101", "Classical Mechanics", "2023-09-01", "2023-12-15", 25, 349.99),
        ("CS 101", "Introduction to Programming", "2023-09-01", "2023-12-15", 40, 399.99),
        ("History 101", "World History", "2023-09-01", "2023-12-15", 35, 279.99)
    ]
    for course in courses:
        app.add_course(*course)
    
    # Create enrollments
    enrollments = [
        (1, 1),  # John in Math 101
        (1, 3),  # John in CS 101
        (2, 1),  # Jane in Math 101
        (2, 2),  # Jane in Physics 101
        (3, 3),  # Mike in CS 101
        (4, 4)   # Sarah in History 101
    ]
    for enrollment in enrollments:
        app.enroll_student(*enrollment)
    
    # Assign teachers to courses
    assignments = [
        (1, 1),  # Alice teaches Math 101
        (2, 2),  # Bob teaches Physics 101
        (3, 3),  # Carol teaches CS 101
        (1, 4)   # Alice teaches History 101
    ]
    for assignment in assignments:
        app.assign_teacher(*assignment)
    
    print("Sample data populated successfully!")
    return app

if __name__ == "__main__":
    app = populate_sample_data()
    
    # Display sample data
    print("\nStudents:")
    for student in app.list_students():
        print(f"ID: {student[0]}, Name: {student[1]} {student[2]}, Email: {student[3]}")
    
    print("\nCourses:")
    for course in app.list_courses():
        print(f"ID: {course[0]}, Title: {course[1]}, Price: ${course[6]:.2f}")