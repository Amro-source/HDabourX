import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-Exercise Ratings (rows=users, columns=exercises)
ratings = np.array([
    [5, 4, 1, 3, 0],  # Alice (0 = ?)
    [3, 0, 5, 2, 4],  # Bob
    [0, 5, 2, 4, 3]   # Carol
])

# Compute user similarities
user_sim = cosine_similarity(ratings)

# Predict Alice's rating for Yoga (index 4)
alice_id = 0
yoga_id = 4

# Find most similar users to Alice (excluding herself)
similar_users = np.argsort(-user_sim[alice_id])[1:]  # [Bob, Carol]

# Take weighted average (using similarities as weights)
numerator = sum(user_sim[alice_id, u] * ratings[u, yoga_id] for u in similar_users)
denominator = sum(user_sim[alice_id, u] for u in similar_users)
predicted_rating = numerator / denominator

print(f"Predicted rating for Alice on Yoga: {predicted_rating:.2f}")