# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:46:44 2024

@author: Meshmesh
"""

import wx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle  # Import pickle for model saving and loading


class MLApp(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(800, 600))
        
        # Set up the GUI layout
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Dataset Loading Section
        dataset_box = wx.BoxSizer(wx.HORIZONTAL)
        dataset_label = wx.StaticText(panel, label="Load Dataset: ")
        self.dataset_path = wx.TextCtrl(panel, size=(400, -1))
        load_button = wx.Button(panel, label="Load")
        load_button.Bind(wx.EVT_BUTTON, self.load_dataset)
        dataset_box.Add(dataset_label, flag=wx.RIGHT, border=8)
        dataset_box.Add(self.dataset_path, proportion=1)
        dataset_box.Add(load_button, flag=wx.LEFT, border=8)
        vbox.Add(dataset_box, flag=wx.EXPAND | wx.ALL, border=10)
        
        # Model Selection Section
        model_box = wx.BoxSizer(wx.HORIZONTAL)
        model_label = wx.StaticText(panel, label="Select Model: ")
        self.model_choice = wx.Choice(panel, choices=["Linear Regression", "Decision Tree", "SVM"])
        self.model_choice.SetSelection(0)
        train_button = wx.Button(panel, label="Train Model")
        train_button.Bind(wx.EVT_BUTTON, self.train_model)
        model_box.Add(model_label, flag=wx.RIGHT, border=8)
        model_box.Add(self.model_choice, flag=wx.RIGHT, border=8)
        model_box.Add(train_button, flag=wx.LEFT, border=8)
        vbox.Add(model_box, flag=wx.EXPAND | wx.ALL, border=10)
        
        # Status and Results Section
        self.status_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 200))
        vbox.Add(self.status_text, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)
        
        # Save Model Section
        save_button = wx.Button(panel, label="Save Model")
        save_button.Bind(wx.EVT_BUTTON, self.save_model)
        vbox.Add(save_button, flag=wx.ALIGN_RIGHT | wx.ALL, border=10)
        
        panel.SetSizer(vbox)
        
        # Internal state
        self.dataset = None
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def load_dataset(self, event):
        with wx.FileDialog(self, "Open CSV file", wildcard="CSV files (*.csv)|*.csv",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return  # The user canceled the file selection
            
            # Get the path of the selected file
            path = file_dialog.GetPath()
            self.dataset_path.SetValue(path)  # Display the path in the text field
            
            try:
                # Load the dataset
                self.dataset = pd.read_csv(path)
                self.status_text.AppendText(f"Dataset loaded successfully from {path}\n")
                self.status_text.AppendText(f"Columns: {', '.join(self.dataset.columns)}\n")
            except Exception as e:
                self.status_text.AppendText(f"Failed to load dataset: {e}\n")

    def train_model(self, event):
        if self.dataset is None:
            self.status_text.AppendText("No dataset loaded. Please load a dataset first.\n")
            return

        target_col = wx.GetTextFromUser("Enter the target column name:", "Target Column")
        if target_col not in self.dataset.columns:
            self.status_text.AppendText(f"Target column '{target_col}' not found in the dataset.\n")
            return

        try:
            # Handle categorical features
            X = self.dataset.drop(columns=[target_col])
            y = self.dataset[target_col]
            
            # Handle missing values in features and target
            self.status_text.AppendText("Checking for missing values...\n")
            X = X.fillna(X.mean(numeric_only=True))  # Fill missing numeric values with the column mean
            X = X.fillna("Missing")  # Fill missing categorical values with "Missing"
            y = y.fillna("Missing")  # Handle missing target values

            # Ensure all non-numeric columns are uniformly strings
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype(str)
            if y.dtype == 'object':  # Ensure target column is uniformly strings if categorical
                y = y.astype(str)

            # Convert categorical columns to numeric
            X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding for features
            if y.dtype == 'object':  # Convert target column if it's categorical
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            # Split dataset
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model_choice = self.model_choice.GetStringSelection()
            if model_choice == "Linear Regression":
                self.model = LinearRegression()
            elif model_choice == "Decision Tree":
                self.model = DecisionTreeClassifier()
            elif model_choice == "SVM":
                self.model = SVC()
            
            self.model.fit(self.x_train, self.y_train)
            self.status_text.AppendText(f"{model_choice} model trained successfully.\n")
            
            # Evaluate the model
            predictions = self.model.predict(self.x_test)
            if model_choice == "Linear Regression":
                mse = mean_squared_error(self.y_test, predictions)
                self.status_text.AppendText(f"Mean Squared Error: {mse}\n")
            else:
                acc = accuracy_score(self.y_test, predictions)
                self.status_text.AppendText(f"Accuracy: {acc}\n")
        except Exception as e:
            self.status_text.AppendText(f"Failed to train model: {e}\n")

    def save_model(self, event):
        if self.model is None:
            self.status_text.AppendText("No trained model to save. Please train a model first.\n")
            return
        
        with wx.FileDialog(self, "Save Model", wildcard="Pickle files (*.pkl)|*.pkl",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as save_dialog:
            if save_dialog.ShowModal() == wx.ID_CANCEL:
                return  # The user canceled the save operation
            
            try:
                path = save_dialog.GetPath()
                # Save the model using Pickle
                with open(path, 'wb') as file:
                    pickle.dump(self.model, file)
                self.status_text.AppendText(f"Model saved to {path}\n")
            except Exception as e:
                self.status_text.AppendText(f"Failed to save model: {e}\n")


if __name__ == "__main__":
    app = wx.App(False)
    frame = MLApp(None, "Machine Learning GUI with wxPython")
    frame.Show()
    app.MainLoop()
