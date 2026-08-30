import json
import pandas as pd
from pathlib import Path

# Load and process HuffPost data with dates
data_path = Path("data/raw/huffpost/News_Category_Dataset_v3.json")
processed_path = Path("data/processed/huffpost_with_dates.csv")

print("Processing HuffPost dataset with dates...")

data = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        data.append(
            {
                "date": row["date"],
                "category": row["category"],
                "headline": row["headline"],
                "text": f"{row['headline']}. {row['short_description']}",
                "link": row["link"],
                "authors": row["authors"],
            }
        )

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

print(f"Loaded {len(df)} articles")
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'Categories: {df["category"].nunique()} unique')
print(f'Sample categories: {df["category"].value_counts().head(10).to_dict()}')

# Save processed data
df.to_csv(processed_path, index=False)
print(f"Saved processed data to {processed_path}")
