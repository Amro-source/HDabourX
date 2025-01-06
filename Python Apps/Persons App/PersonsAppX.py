# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:10:09 2024

@author: M5
"""

import csv
import os
from datetime import datetime

class Person:
    def __init__(self, name, age, weight, person_id, created_at=None):
        self.name = name
        self.age = age
        self.weight = weight
        self.person_id = person_id
        self.created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __str__(self):
        return (f"ID: {self.person_id}, Name: {self.name}, Age: {self.age}, "
                f"Weight: {self.weight}, Created At: {self.created_at}")


class PersonManager:
    def __init__(self, csv_file="persons.csv"):
        self.csv_file = csv_file
        self.persons = []
        self.next_id = 1
        self.load_persons_from_csv()

    def add_person(self, name, age, weight):
        person = Person(name, age, weight, self.next_id)
        self.persons.append(person)
        self.next_id += 1
        print(f"Person added successfully: {person}")

    def delete_person(self, person_id):
        person_to_delete = None
        for person in self.persons:
            if person.person_id == person_id:
                person_to_delete = person
                break
        if person_to_delete:
            self.persons.remove(person_to_delete)
            deletion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Person with ID {person_id} deleted at {deletion_time}.")
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

    def save_persons_to_csv(self):
        """Save the list of persons to a CSV file."""
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["ID", "Name", "Age", "Weight", "Created At"])
            # Write data
            for person in self.persons:
                writer.writerow([person.person_id, person.name, person.age, person.weight, person.created_at])
        print(f"Persons saved to {self.csv_file}")

    def load_persons_from_csv(self):
        """Load the list of persons from a CSV file if it exists."""
        if os.path.exists(self.csv_file):
            with open(self.csv_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    person_id, name, age, weight, created_at = row
                    self.persons.append(Person(name, int(age), float(weight), int(person_id), created_at))
                    self.next_id = max(self.next_id, int(person_id) + 1)
            print(f"Loaded {len(self.persons)} persons from {self.csv_file}")


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

        elif choice == '2':
            person_id = int(input("Enter ID of the person to delete: "))
            manager.delete_person(person_id)

        elif choice == '3':
            print("\nList of Persons:")
            manager.display_persons()

        elif choice == '4':
            manager.total_persons()

        elif choice == '5':
            manager.save_persons_to_csv()
            print("Exiting the program. Data saved.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
