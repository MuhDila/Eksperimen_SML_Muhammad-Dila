import pandas as pd
import numpy as np
import os

# Buat folder output jika belum ada
os.makedirs("dataset_preprocessing", exist_ok=True)

# Load datasets
ratings = pd.read_csv("../dataset_raw/ratings_raw.csv")
books = pd.read_csv("../dataset_raw/ratings_raw.csv", dtype=str)
users = pd.read_csv("../dataset_raw/users_raw.csv")

# === Handling Missing Values & Outliers ===
users = users.dropna(subset=["Age"])
users = users[(users["Age"] >= 5) & (users["Age"] <= 100)]

# Hapus rating = 0
ratings = ratings[ratings["Book-Rating"] != 0]

# Filter user dan buku yang aktif
user_counts = ratings["User-ID"].value_counts()
active_users = user_counts[user_counts >= 3].index
ratings = ratings[ratings["User-ID"].isin(active_users)]

book_counts = ratings["ISBN"].value_counts()
popular_books = book_counts[book_counts >= 3].index
ratings = ratings[ratings["ISBN"].isin(popular_books)]

# === Clean DataFrame for Modelling ===
ratings_clean = ratings[["User-ID", "ISBN", "Book-Rating"]].copy()
ratings_clean.columns = ["user_id", "isbn", "book_rating"]

# Buat user-item matrix (opsional preview)
user_item_matrix = ratings_clean.pivot_table(index="user_id", columns="isbn", values="book_rating")

# === Encoding ===
user_ids = ratings_clean["user_id"].unique().tolist()
isbn_ids = ratings_clean["isbn"].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
isbn_to_isbn_encoded = {x: i for i, x in enumerate(isbn_ids)}

ratings_clean["user"] = ratings_clean["user_id"].map(user_to_user_encoded)
ratings_clean["book"] = ratings_clean["isbn"].map(isbn_to_isbn_encoded)

# Konversi rating jadi float
ratings_clean["book_rating"] = ratings_clean["book_rating"].astype(np.float32)

# === Splitting Dataset ===
ratings_clean = ratings_clean.sample(frac=1, random_state=42)

x = ratings_clean[["user", "book"]].values
y = ratings_clean["book_rating"].apply(lambda x: (x - ratings_clean["book_rating"].min()) / (ratings_clean["book_rating"].max() - ratings_clean["book_rating"].min())).values

train_idx = int(0.8 * len(x))
x_train, x_val = x[:train_idx], x[train_idx:]
y_train, y_val = y[:train_idx], y[train_idx:]

# Simpan hasil preprocessed
ratings_clean.to_csv("dataset_preprocessing/ratings_clean.csv", index=False)
np.save("dataset_preprocessing/x_train.npy", x_train)
np.save("dataset_preprocessing/y_train.npy", y_train)
np.save("dataset_preprocessing/x_val.npy", x_val)
np.save("dataset_preprocessing/y_val.npy", y_val)

print("Preprocessing selesai")
