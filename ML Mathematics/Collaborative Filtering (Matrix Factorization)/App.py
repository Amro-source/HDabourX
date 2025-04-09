# collaborative_filtering.py
from surprise import Dataset, Reader, SVD
import pandas as pd

# Sample data
data = {
    "user": ["Alice", "Alice", "Bob", "Bob", "Carol", "Carol"],
    "exercise": ["Squats", "Push-ups", "Squats", "Running", "Push-ups", "Yoga"],
    "rating": [5, 4, 3, 5, 5, 4]
}
df = pd.DataFrame(data)

# Train model
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[["user", "exercise", "rating"]], reader)
model = SVD()
model.fit(dataset.build_full_trainset())

# Predict
prediction = model.predict("Alice", "Yoga")
print(f"Predicted rating for Alice on Yoga: {prediction.est:.1f}")