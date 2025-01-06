# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:37:24 2024

@author: Meshmesh
"""
import requests

# Send a GET request to the Node.js server
def get_greeting():
    url = "http://localhost:3000/api/greet"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print(data['message'])  # Output: Hello from Node.js!
    else:
        print(f"Failed to get response: {response.status_code}")

# Send a POST request to the Node.js server with data
def send_data(name, age):
    url = "http://localhost:3000/api/data"
    payload = {"name": name, "age": age}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(data['message'])  # Output: Hello [name], you are [age] years old.
    else:
        print(f"Failed to post data: {response.status_code}")

if __name__ == "__main__":
    get_greeting()  # Call the GET endpoint
    send_data("Alice", 30)  # Call the POST endpoint with data

