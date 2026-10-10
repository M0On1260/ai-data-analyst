"""Build eval/benchmark.json for data/sales.csv.

Expected answers are computed directly with pandas, so the ground truth is
correct by construction. Add your own questions to make the benchmark harder.
"""
import json
from pathlib import Path

import pandas as pd

df = pd.read_csv("data/sales.csv")
df["order_date"] = pd.to_datetime(df["order_date"])
df["year"] = df["order_date"].dt.year


def num(q, value, tol=0.01):
    return {"question": q, "type": "number", "expected": float(value), "tolerance": tol}


def text(q, *answers):
    return {"question": q, "type": "string", "expected": list(answers)}


rev_by_year = df.groupby("year")["revenue"].sum()
rev_2025_month = df[df.year == 2025].groupby(df.order_date.dt.month)["revenue"].sum()
best_month = int(rev_2025_month.idxmax())
month_names = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

items = [
    num("How many orders are in the dataset?", len(df), 0),
    num("What is the total revenue?", df.revenue.sum()),
    text("Which region has the highest total revenue?", df.groupby("region").revenue.sum().idxmax()),
    text("Which category has the lowest total revenue?", df.groupby("category").revenue.sum().idxmin()),
    num("What is the average revenue per order?", df.revenue.mean()),
    num("How many unique products are sold?", df["product"].nunique(), 0),
    num("What is the total revenue from the Electronics category?", df[df.category == "Electronics"].revenue.sum()),
    text("Which product sold the most units in total?", df.groupby("product").units.sum().idxmax()),
    text("In which month of 2025 was total revenue the highest? Answer with the month name.",
         month_names[best_month - 1], f"2025-{best_month:02d}"),
    num("What percentage of orders are from the Corporate segment?", (df.customer_segment == "Corporate").mean() * 100),
    num("What is the average discount_pct, ignoring missing values?", df.discount_pct.mean()),
    num("How many orders have a missing discount_pct?", df.discount_pct.isna().sum(), 0),
    num("What is the total revenue in 2024?", rev_by_year[2024]),
    text("Which region has the highest average revenue per order?", df.groupby("region").revenue.mean().idxmax()),
    num("What is the median unit_price in the Furniture category?", df[df.category == "Furniture"].unit_price.median()),
    num("How many orders had more than 5 units?", (df.units > 5).sum(), 0),
    num("By what percentage did total revenue grow from 2024 to 2025?", (rev_by_year[2025] / rev_by_year[2024] - 1) * 100),
    text("Which customer_segment has the highest average units per order?", df.groupby("customer_segment").units.mean().idxmax()),
    num("What is the highest single-order revenue?", df.revenue.max()),
    num("How many orders were placed in the South region in the fourth quarter of 2025?",
        ((df.region == "South") & (df.year == 2025) & (df.order_date.dt.quarter == 4)).sum(), 0),
]

Path("eval").mkdir(exist_ok=True)
Path("eval/benchmark.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
print(f"Wrote eval/benchmark.json with {len(items)} questions")
