"""Generate a reproducible synthetic sales dataset at data/sales.csv.

Swap this for a real dataset (VTU results, IPL stats, government open data)
before putting the project on your resume.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

CATEGORIES = {
    "Electronics": (["Laptop", "Headphones", "Smartwatch", "Tablet"], 8000, 60000),
    "Furniture": (["Office Chair", "Study Desk", "Bookshelf", "Sofa"], 3000, 25000),
    "Clothing": (["T-Shirt", "Jeans", "Jacket", "Sneakers"], 500, 4000),
    "Books": (["Novel", "Textbook", "Comic", "Notebook"], 150, 900),
    "Groceries": (["Rice 5kg", "Olive Oil", "Coffee", "Snack Box"], 50, 600),
}
REGIONS = ["North", "South", "East", "West"]
REGION_WEIGHTS = [0.22, 0.38, 0.17, 0.23]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
SEGMENT_WEIGHTS = [0.55, 0.20, 0.25]

cat_names = list(CATEGORIES)
categories = rng.choice(cat_names, N, p=[0.15, 0.15, 0.25, 0.25, 0.20])
products, prices = [], []
for c in categories:
    names, lo, hi = CATEGORIES[c]
    products.append(rng.choice(names))
    prices.append(round(float(rng.uniform(lo, hi)), 2))

dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 731, N), unit="D")
segments = rng.choice(SEGMENTS, N, p=SEGMENT_WEIGHTS)
units = np.where(segments == "Corporate", rng.integers(2, 12, N), rng.integers(1, 7, N))
discount = rng.choice([0, 5, 10, 15, 20], N, p=[0.4, 0.2, 0.2, 0.15, 0.05]).astype(float)

df = pd.DataFrame({
    "order_id": np.arange(1001, 1001 + N),
    "order_date": dates.strftime("%Y-%m-%d"),
    "region": rng.choice(REGIONS, N, p=REGION_WEIGHTS),
    "category": categories,
    "product": products,
    "customer_segment": segments,
    "units": units,
    "unit_price": prices,
    "discount_pct": discount,
})
df["revenue"] = (df["units"] * df["unit_price"] * (1 - df["discount_pct"] / 100)).round(2)

# Realistic mess: some discounts were never recorded.
df.loc[rng.choice(N, 40, replace=False), "discount_pct"] = np.nan

Path("data").mkdir(exist_ok=True)
df.to_csv("data/sales.csv", index=False)
print(f"Wrote data/sales.csv with {len(df)} rows")
