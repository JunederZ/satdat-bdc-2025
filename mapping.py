import pandas as pd

# Load dataset
df = pd.read_csv("datatrain.csv")

# Normalize text to lowercase & strip spaces
df["emotion_clean"] = df["emotion"].str.lower().str.strip()

# Apply mapping
from label_mapping import label_map
df["emotion_normalized"] = df["emotion_clean"].map(label_map)

# Drop rows with unmapped labels (if any)
df = df.dropna(subset=["emotion_normalized"])

# Save the cleaned dataset
df.to_csv("datatrain_clean.csv", index=False)

print("Cleaned dataset saved → datatrain_clean.csv")
print("Unique labels after cleaning:", df["emotion_normalized"].unique())
