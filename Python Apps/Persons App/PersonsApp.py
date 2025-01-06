# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

class Person:
    def __init__(self, name, age, weight, person_id):
        self.name = name
        self.age = age
        self.weight = weight
        self.person_id = person_id

    def __str__(self):
        return f"ID: {self.person_id}, Name: {self.name}, Age: {self.age}, Weight: {self.weight}"

class PersonManager:
    def __init__(self):
        self.persons = []
        self.next_id = 1  # The next ID to assign to a person

    def add_person(self, name, age, weight):
        person = Person(name, age, weight, self.next_id)
        self.persons.append(person)
        self.next_id += 1

    def delete_person(self, person_id):
        person_to_delete = None
        for person in self.persons:
            if person.person_id == person_id:
                person_to_delete = person
                break
        if person_to_delete:
            self.persons.remove(person_to_delete)
            print(f"Person with ID {person_id} deleted.")
        else:
            print(f"No person found with ID {person_id}")

    def display_persons(self):
        if not self.persons:
            print("No persons available.")
        else:
            for person in self.persons:
                print(person)

    def total_persons(self):
        print(f"Total number of persons: {len(self.persons)}")

def main():
    manager = PersonManager()

    while True:
        print("\nMenu:")
        print("1. Add Person")
        print("2. Delete Person")
        print("3. Display All Persons")
        print("4. Total Number of Persons")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            weight = float(input("Enter weight: "))
            manager.add_person(name, age, weight)
            print("Person added successfully.")

        elif choice == '2':
            person_id = int(input("Enter ID of the person to delete: "))
            manager.delete_person(person_id)

        elif choice == '3':
            print("\nList of Persons:")
            manager.display_persons()

        elif choice == '4':
            manager.total_persons()

        elif choice == '5':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
