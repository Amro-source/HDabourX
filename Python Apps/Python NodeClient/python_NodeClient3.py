# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:24:55 2024

@author: Meshmesh
"""

import requests

# Send a POST request to the Node.js server with data
def send_data(name, age):
    url = "http://localhost:3000/api/data"
    payload = {"name": name, "age": age}
    
    response = requests.post(url, json=payload)  # POST request
    if response.status_code == 200:
        data = response.json()
        print(data['message'])  # Output: Hello [name], you are [age] years old.
    else:
        print(f"Failed to post data: {response.status_code} - {response.text}")

if __name__ == "__main__":
    send_data("Alice", 30)
