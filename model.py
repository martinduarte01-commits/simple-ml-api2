import pandas as pd
import joblib

from pathlib import Path
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------

orig_url = "https://docs.google.com/spreadsheets/d/1mlXah6ADb88FI6bL0pIQap0gxD7IoimT/edit?usp=sharing"

file_id = orig_url.split("/")[-2]

data_path = (
    "https://drive.google.com/uc?export=download&id="
    + file_id
)


# ---------------------------------------------------------
# 2. Load offers
# ---------------------------------------------------------

df_offers = pd.read_excel(
    data_path,
    sheet_name=0
)

df_offers = df_offers.iloc[:, :6]

df_offers.columns = [
    "offer_id",
    "campaign",
    "wine_variety",
    "min_qty",
    "discount",
    "origin"
]


# ---------------------------------------------------------
# 3. Load transactions
# ---------------------------------------------------------

df_transactions = pd.read_excel(
    data_path,
    sheet_name=1
)

df_transactions.columns = [
    "customer_name",
    "offer_id"
]

df_transactions["frequency"] = 1


# ---------------------------------------------------------
# 4. Merge offers and transactions
# ---------------------------------------------------------

df_merged = df_offers.merge(
    df_transactions,
    how="left",
    on="offer_id"
)


# ---------------------------------------------------------
# 5. Create customer x offer matrix
# ---------------------------------------------------------

pivot = pd.pivot_table(
    df_merged,
    index="customer_name",
    columns="offer_id",
    values="frequency",
    aggfunc="sum",
    fill_value=0
)


# Data used to train KMeans
X = pivot.values


print("Training data shape:", X.shape)
print("Number of customers:", X.shape[0])
print("Number of offers:", X.shape[1])


# ---------------------------------------------------------
# 6. Train final KMeans model
# ---------------------------------------------------------

K = 3

model = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=100
)

model.fit(X)


# ---------------------------------------------------------
# 7. Save model AND feature information
# ---------------------------------------------------------

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

model_bundle = {
    "model": model,
    "offer_ids": pivot.columns.tolist()
}

joblib.dump(
    model_bundle,
    "models/model.joblib"
)


print("Model training complete.")
print("Model saved to models/model.joblib")
print("Offer IDs:", pivot.columns.tolist())
