from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_FILE = PROJECT_DIR / "retail_orders_raw.csv"
CLEAN_FILE = PROJECT_DIR / "retail_orders_clean.csv"
CHART_DIR = PROJECT_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)


def make_chart(title, labels, values):
    width = 760
    height = 360
    left = 190
    top = 48
    chart_width = 520
    chart_height = 250
    largest_value = max(values) if values else 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-family="Arial" font-size="20" font-weight="bold">{title}</text>',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * (chart_height / max(1, len(labels)))
        bar_width = chart_width * value / largest_value
        parts.extend([
            f'<text x="{left - 10}" y="{y + 16:.1f}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>',
            f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" height="22" rx="4" fill="#276EF1"/>',
            f'<text x="{left + bar_width + 8:.1f}" y="{y + 16:.1f}" font-family="Arial" font-size="12">{value:,.0f}</text>',
        ])

    parts.append("</svg>")
    return "\n".join(parts)


def create_sample_data():
    random = np.random.default_rng(42)
    row_count = 420
    order_dates = pd.date_range("2024-01-01", periods=180, freq="D")
    categories = ["Accessories", "Books", "Home", "Electronics"]

    data = pd.DataFrame({
        "order_id": [f"O{number:04d}" for number in range(row_count)],
        "order_date": random.choice(order_dates, row_count),
        "region": random.choice(["North", "South", "East", "West"], row_count),
        "category": random.choice(categories, row_count, p=[0.28, 0.18, 0.32, 0.22]),
        "quantity": random.integers(1, 8, row_count),
        "unit_price": np.round(random.lognormal(3.1, 0.65, row_count), 2),
        "customer_rating": np.round(random.normal(4.0, 0.75, row_count).clip(1, 5), 1),
    })
    data["revenue"] = (data["quantity"] * data["unit_price"]).round(2)

    data.loc[random.choice(row_count, 24, replace=False), "customer_rating"] = np.nan
    data.loc[random.choice(row_count, 16, replace=False), "region"] = np.nan
    data.loc[random.choice(row_count, 5, replace=False), "quantity"] = -1
    data.loc[random.choice(row_count, 4, replace=False), "unit_price"] = 0
    data.loc[random.choice(row_count, 3, replace=False), "unit_price"] *= 18
    data = pd.concat([data, data.iloc[[7, 32, 95]]], ignore_index=True)
    data.to_csv(RAW_FILE, index=False)
    return data


def clean_data(data):
    duplicate_count = int(data.duplicated(subset="order_id").sum())
    missing_values = data.isna().sum()

    cleaned = data.drop_duplicates(subset="order_id", keep="first").copy()
    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] > 0)].copy()
    cleaned["region"] = cleaned["region"].fillna("Unknown")
    cleaned["customer_rating"] = cleaned["customer_rating"].fillna(
        cleaned["customer_rating"].median()
    )

    first_quartile = cleaned["unit_price"].quantile(0.25)
    third_quartile = cleaned["unit_price"].quantile(0.75)
    price_range = third_quartile - first_quartile
    lower_limit = first_quartile - 1.5 * price_range
    upper_limit = third_quartile + 1.5 * price_range
    outlier_count = int(
        ((cleaned["unit_price"] < lower_limit) | (cleaned["unit_price"] > upper_limit)).sum()
    )

    cleaned["unit_price"] = cleaned["unit_price"].clip(lower_limit, upper_limit)
    cleaned["revenue"] = (cleaned["quantity"] * cleaned["unit_price"]).round(2)
    cleaned = cleaned.sort_values("order_date").reset_index(drop=True)
    return cleaned, duplicate_count, missing_values, outlier_count


def write_report(raw_row_count, cleaned, duplicate_count, missing_values, outlier_count):
    monthly_revenue = cleaned.groupby(cleaned["order_date"].dt.to_period("M"))["revenue"].sum()
    category_revenue = cleaned.groupby("category")["revenue"].sum().sort_values(ascending=False)
    total_revenue = cleaned["revenue"].sum()

    charts = {
        "monthly_revenue.svg": make_chart(
            "Monthly revenue after cleaning",
            [str(month) for month in monthly_revenue.index],
            monthly_revenue.tolist(),
        ),
        "revenue_by_category.svg": make_chart(
            "Revenue by product category",
            list(category_revenue.index),
            category_revenue.tolist(),
        ),
    }
    price_counts = pd.cut(cleaned["unit_price"], bins=8).value_counts().sort_index()
    charts["price_distribution.svg"] = make_chart(
        "Capped unit price distribution",
        [f"{interval.left:.0f}–{interval.right:.0f}" for interval in price_counts.index],
        price_counts.tolist(),
    )
    for filename, chart in charts.items():
        (CHART_DIR / filename).write_text(chart, encoding="utf-8")

    report = f'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>Retail Data Cleaning Report</title>
<style>
body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 980px; margin: 40px auto; padding: 0 22px; color: #172033; }}
h1, h2 {{ color: #15346b; }}
.note {{ background: #fff5da; padding: 14px; border-left: 4px solid #d99b00; }}
.stats {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.stat {{ background: #f1f5fb; padding: 14px 18px; border-radius: 8px; min-width: 150px; }}
.stat b {{ display: block; font-size: 1.4rem; }}
img {{ width: min(100%, 760px); display: block; margin: 18px 0; }}
</style>
<h1>Retail orders: cleaning &amp; exploration</h1>
<p class="note"><b>Dataset:</b> This is synthetic sample data generated by the script. It does not contain real customer or business information, and the currency is unspecified.</p>
<div class="stats">
<div class="stat">Raw rows<b>{raw_row_count}</b></div>
<div class="stat">Clean rows<b>{len(cleaned)}</b></div>
<div class="stat">Duplicate order IDs removed<b>{duplicate_count}</b></div>
<div class="stat">Price outliers capped<b>{outlier_count}</b></div>
<div class="stat">Clean revenue<b>{total_revenue:,.0f}</b></div>
</div>
<h2>Cleaning steps</h2>
<ol>
<li>Removed repeated order IDs, keeping the first row.</li>
<li>Removed rows with a zero or negative quantity or price.</li>
<li>Filled missing regions with “Unknown” and missing ratings with the median.</li>
<li>Capped unusually high or low prices at the 1.5×IQR limits.</li>
<li>Recalculated revenue using the cleaned quantity and price.</li>
</ol>
<p>The raw file had {int(missing_values.sum())} missing cells: {int(missing_values.get('region', 0))} region values and {int(missing_values.get('customer_rating', 0))} ratings.</p>
<h2>Findings</h2>
<p>Total revenue in the cleaned sample is {total_revenue:,.2f}. {category_revenue.index[0]} had the highest category revenue ({category_revenue.iloc[0]:,.2f}). The month with the highest revenue was {monthly_revenue.idxmax()} ({monthly_revenue.max():,.2f}).</p>
<img src="charts/monthly_revenue.svg" alt="Monthly revenue bar chart">
<img src="charts/revenue_by_category.svg" alt="Revenue by category bar chart">
<img src="charts/price_distribution.svg" alt="Distribution of capped unit prices">
<h2>Run the script</h2>
<p>Install pandas and numpy, then run <code>python cleaning_project.py</code>. The script recreates the CSV files, charts, and this report.</p>
<ul>
<li><a href="retail_orders_raw.csv">Raw sample data</a></li>
<li><a href="retail_orders_clean.csv">Cleaned data</a></li>
<li><a href="cleaning_project.py">Python script</a></li>
</ul>
</html>'''
    (PROJECT_DIR / "report.html").write_text(report, encoding="utf-8")


def main():
    raw_data = create_sample_data()
    cleaned_data, duplicate_count, missing_values, outlier_count = clean_data(raw_data)
    cleaned_data.to_csv(CLEAN_FILE, index=False, date_format="%Y-%m-%d")
    write_report(
        len(raw_data), cleaned_data, duplicate_count, missing_values, outlier_count
    )
    print(f"Finished: {len(raw_data)} raw rows, {len(cleaned_data)} cleaned rows.")


if __name__ == "__main__":
    main()
