from surprise import Dataset, Reader, SVD
import pandas as pd

# 1. Create sample exercise rating data
data = {
    'user': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob', 'Carol', 'Carol', 'Carol'],
    'exercise': ['Squats', 'Push-ups', 'Running', 'Squats', 'Running', 'Yoga', 'Push-ups', 'Running', 'Yoga'],
    'rating': [5, 4, 1, 3, 5, 2, 5, 2, 4]
}

# 2. Convert to Pandas DataFrame
df = pd.DataFrame(data)

# 3. Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user', 'exercise', 'rating']], reader)

# 4. Build and train model
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# 5. Make a prediction (Alice's rating for Yoga)
pred = algo.predict('Alice', 'Yoga')
print(f"Predicted rating for Alice on Yoga: {pred.est:.1f}")