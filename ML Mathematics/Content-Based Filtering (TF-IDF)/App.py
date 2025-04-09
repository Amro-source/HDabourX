# content_based.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample data
data = {
    "exercise": ["Squats", "Push-ups", "Running", "Yoga"],
    "description": [
        "legs strength high-intensity",
        "upper-body strength high-intensity",
        "cardio endurance low-intensity",
        "flexibility mindfulness low-intensity"
    ]
}
df = pd.DataFrame(data)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["description"])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get recommendations for "Squats"
exercise_idx = df[df["exercise"] == "Squats"].index[0]
similar_scores = list(enumerate(cosine_sim[exercise_idx]))
similar_exercises = sorted(similar_scores, key=lambda x: x[1], reverse=True)[1:3]  # Top 2 matches

print("Exercises similar to Squats:")
for idx, score in similar_exercises:
    print(f"- {df.iloc[idx]['exercise']} (Score: {score:.2f})")