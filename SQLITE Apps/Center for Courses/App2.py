import sqlite3
from datetime import datetime

def initialize_database():
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
    
    # Junction table for many-to-many relationship between students and courses
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
    
    # Junction table for many-to-many relationship between teachers and courses
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
    
    def __del__(self):
        self.conn.close()
    
    # Student operations
    def add_student(self, first_name, last_name, email, phone):
        try:
            self.cursor.execute('''
            INSERT INTO students (first_name, last_name, email, phone)
            VALUES (?, ?, ?, ?)
            ''', (first_name, last_name, email, phone))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            print("Error: Email already exists or invalid data.")
            return False
    
    def list_students(self):
        self.cursor.execute('SELECT * FROM students')
        return self.cursor.fetchall()
    
    # Teacher operations
    def add_teacher(self, first_name, last_name, email, phone, specialization):
        try:
            self.cursor.execute('''
            INSERT INTO teachers (first_name, last_name, email, phone, specialization)
            VALUES (?, ?, ?, ?, ?)
            ''', (first_name, last_name, email, phone, specialization))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            print("Error: Email already exists or invalid data.")
            return False
    
    def list_teachers(self):
        self.cursor.execute('SELECT * FROM teachers')
        return self.cursor.fetchall()
    
    # Course operations
    def add_course(self, title, description, start_date, end_date, capacity, price):
        try:
            self.cursor.execute('''
            INSERT INTO courses (title, description, start_date, end_date, capacity, price)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (title, description, start_date, end_date, capacity, price))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error adding course: {e}")
            return False
    
    def list_courses(self):
        self.cursor.execute('SELECT * FROM courses')
        return self.cursor.fetchall()
    
    # Enrollment operations
    def enroll_student(self, student_id, course_id):
        try:
            self.cursor.execute('''
            INSERT INTO student_courses (student_id, course_id)
            VALUES (?, ?)
            ''', (student_id, course_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error enrolling student: {e}")
            return False
    
    def assign_teacher(self, teacher_id, course_id):
        try:
            self.cursor.execute('''
            INSERT INTO teacher_courses (teacher_id, course_id)
            VALUES (?, ?)
            ''', (teacher_id, course_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error assigning teacher: {e}")
            return False
    
    # Reporting functions
    def get_student_courses(self, student_id):
        self.cursor.execute('''
        SELECT c.course_id, c.title, c.start_date, c.end_date
        FROM courses c
        JOIN student_courses sc ON c.course_id = sc.course_id
        WHERE sc.student_id = ?
        ''', (student_id,))
        return self.cursor.fetchall()
    
    def get_course_students(self, course_id):
        self.cursor.execute('''
        SELECT s.student_id, s.first_name, s.last_name, s.email
        FROM students s
        JOIN student_courses sc ON s.student_id = sc.student_id
        WHERE sc.course_id = ?
        ''', (course_id,))
        return self.cursor.fetchall()
    
    def get_teacher_courses(self, teacher_id):
        self.cursor.execute('''
        SELECT c.course_id, c.title, c.start_date, c.end_date
        FROM courses c
        JOIN teacher_courses tc ON c.course_id = tc.course_id
        WHERE tc.teacher_id = ?
        ''', (teacher_id,))
        return self.cursor.fetchall()
    
    def get_course_teachers(self, course_id):
        self.cursor.execute('''
        SELECT t.teacher_id, t.first_name, t.last_name, t.specialization
        FROM teachers t
        JOIN teacher_courses tc ON t.teacher_id = tc.teacher_id
        WHERE tc.course_id = ?
        ''', (course_id,))
        return self.cursor.fetchall()




def display_menu():
    print("\nCourse Center Management System")
    print("1. Manage Students")
    print("2. Manage Teachers")
    print("3. Manage Courses")
    print("4. Enrollment Management")
    print("5. Reports")
    print("6. Exit")

def manage_students(app):
    while True:
        print("\nStudent Management")
        print("1. Add Student")
        print("2. List Students")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            first_name = input("First Name: ")
            last_name = input("Last Name: ")
            email = input("Email: ")
            phone = input("Phone: ")
            if app.add_student(first_name, last_name, email, phone):
                print("Student added successfully!")
        
        elif choice == '2':
            students = app.list_students()
            for student in students:
                print(f"ID: {student[0]}, Name: {student[1]} {student[2]}, Email: {student[3]}, Phone: {student[4]}")
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice. Please try again.")

def manage_teachers(app):
    while True:
        print("\nTeacher Management")
        print("1. Add Teacher")
        print("2. List Teachers")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            first_name = input("First Name: ")
            last_name = input("Last Name: ")
            email = input("Email: ")
            phone = input("Phone: ")
            specialization = input("Specialization: ")
            if app.add_teacher(first_name, last_name, email, phone, specialization):
                print("Teacher added successfully!")
        
        elif choice == '2':
            teachers = app.list_teachers()
            for teacher in teachers:
                print(f"ID: {teacher[0]}, Name: {teacher[1]} {teacher[2]}, Email: {teacher[3]}, Specialization: {teacher[5]}")
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice. Please try again.")

def manage_courses(app):
    while True:
        print("\nCourse Management")
        print("1. Add Course")
        print("2. List Courses")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            title = input("Course Title: ")
            description = input("Description: ")
            start_date = input("Start Date (YYYY-MM-DD): ")
            end_date = input("End Date (YYYY-MM-DD): ")
            capacity = int(input("Capacity: "))
            price = float(input("Price: "))
            if app.add_course(title, description, start_date, end_date, capacity, price):
                print("Course added successfully!")
        
        elif choice == '2':
            courses = app.list_courses()
            for course in courses:
                print(f"ID: {course[0]}, Title: {course[1]}, Dates: {course[3]} to {course[4]}, Price: {course[6]}")
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice. Please try again.")

def manage_enrollments(app):
    while True:
        print("\nEnrollment Management")
        print("1. Enroll Student in Course")
        print("2. Assign Teacher to Course")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            while True:
                student_id = input("Student ID (or 'back' to return): ")
                if student_id.lower() == 'back':
                    break
                
                if not student_id:
                    print("Error: Student ID cannot be empty")
                    continue
                
                try:
                    student_id = int(student_id)
                except ValueError:
                    print("Error: Please enter a valid number for Student ID")
                    continue
                
                # List available courses
                print("\nAvailable Courses:")
                courses = app.list_courses()
                for course in courses:
                    print(f"ID: {course[0]}, Title: {course[1]}")
                
                while True:
                    course_id = input("\nCourse ID (or 'back' to return): ")
                    if course_id.lower() == 'back':
                        break
                    
                    if not course_id:
                        print("Error: Course ID cannot be empty")
                        continue
                    
                    try:
                        course_id = int(course_id)
                    except ValueError:
                        print("Error: Please enter a valid number for Course ID")
                        continue
                    
                    # Try to enroll the student
                    if app.enroll_student(student_id, course_id):
                        print("Student enrolled successfully!")
                        break
                    else:
                        print("Failed to enroll student. Please check the IDs and try again.")
                break
        
        elif choice == '2':
            while True:
                teacher_id = input("Teacher ID (or 'back' to return): ")
                if teacher_id.lower() == 'back':
                    break
                
                if not teacher_id:
                    print("Error: Teacher ID cannot be empty")
                    continue
                
                try:
                    teacher_id = int(teacher_id)
                except ValueError:
                    print("Error: Please enter a valid number for Teacher ID")
                    continue
                
                # List available courses
                print("\nAvailable Courses:")
                courses = app.list_courses()
                for course in courses:
                    print(f"ID: {course[0]}, Title: {course[1]}")
                
                while True:
                    course_id = input("\nCourse ID (or 'back' to return): ")
                    if course_id.lower() == 'back':
                        break
                    
                    if not course_id:
                        print("Error: Course ID cannot be empty")
                        continue
                    
                    try:
                        course_id = int(course_id)
                    except ValueError:
                        print("Error: Please enter a valid number for Course ID")
                        continue
                    
                    # Try to assign the teacher
                    if app.assign_teacher(teacher_id, course_id):
                        print("Teacher assigned successfully!")
                        break
                    else:
                        print("Failed to assign teacher. Please check the IDs and try again.")
                break
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice. Please try again.")

def view_reports(app):
    while True:
        print("\nReports Dashboard")
        print("1. Generate All Student Enrollments Report")
        print("2. Generate All Course Assignments Report")
        print("3. Generate Teacher Workload Report")
        print("4. Generate Course Popularity Report")
        print("5. Back to Main Menu")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            print("\n=== All Student Enrollments ===\n")
            students = app.list_students()
            for student in students:
                print(f"\nStudent: {student[1]} {student[2]} (ID: {student[0]})")
                courses = app.get_student_courses(student[0])
                if courses:
                    for course in courses:
                        print(f"  - {course[1]} (Course ID: {course[0]})")
                else:
                    print("  No courses enrolled")
            print("\n=== End of Report ===\n")
        
        elif choice == '2':
            print("\n=== All Course Assignments ===\n")
            courses = app.list_courses()
            for course in courses:
                print(f"\nCourse: {course[1]} (ID: {course[0]})")
                # Get students
                students = app.get_course_students(course[0])
                print(f"  Students enrolled: {len(students)}")
                # Get teachers
                teachers = app.get_course_teachers(course[0])
                if teachers:
                    for teacher in teachers:
                        print(f"  - Taught by: {teacher[1]} {teacher[2]} (Specialization: {teacher[3]})")
                else:
                    print("  No teachers assigned")
            print("\n=== End of Report ===\n")
        
        elif choice == '3':
            print("\n=== Teacher Workload Report ===\n")
            teachers = app.list_teachers()
            for teacher in teachers:
                courses = app.get_teacher_courses(teacher[0])
                print(f"\nTeacher: {teacher[1]} {teacher[2]} (ID: {teacher[0]})")
                print(f"Specialization: {teacher[5]}")
                print(f"Number of courses assigned: {len(courses)}")
                if courses:
                    for course in courses:
                        print(f"  - {course[1]} (Course ID: {course[0]})")
            print("\n=== End of Report ===\n")
        
        elif choice == '4':
            print("\n=== Course Popularity Report ===\n")
            courses = app.list_courses()
            # Create a list with course ID, title, and student count
            popularity = []
            for course in courses:
                students = app.get_course_students(course[0])
                popularity.append((course[0], course[1], len(students), course[6]))
            
            # Sort by number of students (descending)
            popularity.sort(key=lambda x: x[2], reverse=True)
            
            print("\nCourses by Popularity (student count):\n")
            for course in popularity:
                print(f"{course[1]} (ID: {course[0]}):")
                print(f"  - Students enrolled: {course[2]}")
                print(f"  - Price: ${course[3]:.2f}")
                print(f"  - Fill rate: {(course[2]/course[3])*100:.1f}% of capacity\n" if course[3] > 0 else "  - Capacity: Unlimited\n")
            
            print("\n=== End of Report ===\n")
        
        elif choice == '5':
            break
        
        else:
            print("Invalid choice. Please try again.")

def main():
    initialize_database()
    app = CourseCenterApp()
    
    while True:
        display_menu()
        choice = input("Enter your choice: ")
        
        if choice == '1':
            manage_students(app)
        elif choice == '2':
            manage_teachers(app)
        elif choice == '3':
            manage_courses(app)
        elif choice == '4':
            manage_enrollments(app)
        elif choice == '5':
            view_reports(app)
        elif choice == '6':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

    