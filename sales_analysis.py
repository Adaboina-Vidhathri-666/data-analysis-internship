"""
============================================================
  INTERNSHIP PROJECT: Sales Data Analysis using Pandas
  Author  : [Your Name]
  Tool    : Python (Pandas, Matplotlib, Seaborn)
  Dataset : sales_data.csv (Retail Sales - Jan to Jul 2024)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  STEP 1: LOADING DATASET")
print("=" * 55)

df = pd.read_csv('sales_data.csv')

print(f"\n✅ Dataset loaded successfully!")
print(f"   Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Columns:\n   {list(df.columns)}")
print(f"\n🔍 First 5 rows:")
print(df.head())

# ─────────────────────────────────────────────
# STEP 2: DATA CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2: DATA CLEANING")
print("=" * 55)

# 2a. Check missing values
print("\n🚨 Missing values before cleaning:")
print(df.isnull().sum())

# 2b. Drop rows where Date or Category is missing (critical columns)
df.dropna(subset=['Date', 'Category'], inplace=True)

# 2c. Fill missing Customer_Rating with median
median_rating = df['Customer_Rating'].median()
df['Customer_Rating'].fillna(median_rating, inplace=True)

print(f"\n✅ After cleaning — Shape: {df.shape}")
print(f"   Remaining nulls: {df.isnull().sum().sum()}")

# 2d. Parse dates
df['Date'] = pd.to_datetime(df['Date'])
df['Month']  = df['Date'].dt.month_name()
df['Month_Num'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)

# 2e. Remove duplicates
dupes_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"   Duplicate rows removed: {dupes_before}")

# 2f. Check data types
print("\n📊 Data Types:")
print(df.dtypes)

# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 3: FEATURE ENGINEERING")
print("=" * 55)

# Revenue = Quantity × Unit_Price × (1 - Discount)
df['Revenue'] = df['Quantity'] * df['Unit_Price'] * (1 - df['Discount'])
df['Revenue'] = df['Revenue'].round(2)

print("\n✅ New column 'Revenue' created.")
print(f"   Sample revenues: {df['Revenue'].head(5).tolist()}")

# ─────────────────────────────────────────────
# STEP 4: SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4: SUMMARY STATISTICS")
print("=" * 55)

print("\n📈 Descriptive Statistics (Numeric Columns):")
print(df[['Quantity', 'Unit_Price', 'Discount', 'Revenue', 'Customer_Rating']].describe().round(2))

# ─────────────────────────────────────────────
# STEP 5: GROUPING & AGGREGATION
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 5: GROUPING & AGGREGATION")
print("=" * 55)

# 5a. Revenue by Category
rev_by_cat = df.groupby('Category')['Revenue'].agg(['sum', 'mean', 'count']).round(2)
rev_by_cat.columns = ['Total_Revenue', 'Avg_Revenue', 'Orders']
rev_by_cat.sort_values('Total_Revenue', ascending=False, inplace=True)
print("\n🏷️  Revenue by Category:")
print(rev_by_cat)

# 5b. Revenue by Region
rev_by_region = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False).round(2)
print("\n🌍 Revenue by Region:")
print(rev_by_region)

# 5c. Sales Rep Performance
rep_perf = df.groupby('Sales_Rep').agg(
    Total_Revenue=('Revenue', 'sum'),
    Orders=('Order_ID', 'count'),
    Avg_Rating=('Customer_Rating', 'mean')
).round(2).sort_values('Total_Revenue', ascending=False)
print("\n👤 Sales Rep Performance:")
print(rep_perf)

# 5d. Monthly Revenue Trend
monthly_rev = df.groupby('Month_Num').agg(
    Month=('Month', 'first'),
    Revenue=('Revenue', 'sum')
).reset_index().sort_values('Month_Num')
print("\n📅 Monthly Revenue Trend:")
print(monthly_rev[['Month', 'Revenue']])

# 5e. Top 5 Products by Revenue
top_products = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(5)
print("\n🏆 Top 5 Products by Revenue:")
print(top_products)

# ─────────────────────────────────────────────
# STEP 6: FILTERING EXAMPLES
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 6: FILTERING")
print("=" * 55)

# High-value orders (Revenue > 100,000)
high_value = df[df['Revenue'] > 100_000]
print(f"\n💰 High-value orders (Revenue > ₹1,00,000): {len(high_value)}")
print(high_value[['Order_ID', 'Product', 'Revenue', 'Region']].to_string(index=False))

# Top-rated transactions (Rating >= 4.7)
top_rated = df[df['Customer_Rating'] >= 4.7]
print(f"\n⭐ High-rated orders (Rating ≥ 4.7): {len(top_rated)}")
print(top_rated[['Order_ID', 'Product', 'Category', 'Customer_Rating']].to_string(index=False))

# Electronics in Q1
elec_q1 = df[(df['Category'] == 'Electronics') & (df['Quarter'] == '2024Q1')]
print(f"\n📱 Electronics orders in Q1 2024: {len(elec_q1)}")

# ─────────────────────────────────────────────
# STEP 7: KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 7: KEY INSIGHTS")
print("=" * 55)

total_rev    = df['Revenue'].sum()
total_orders = df.shape[0]
best_cat     = rev_by_cat['Total_Revenue'].idxmax()
best_region  = rev_by_region.idxmax()
best_rep     = rep_perf['Total_Revenue'].idxmax()
avg_discount = df['Discount'].mean() * 100

print(f"""
  ✅ Total Revenue     : ₹{total_rev:,.2f}
  ✅ Total Orders      : {total_orders}
  ✅ Best Category     : {best_cat}  (₹{rev_by_cat.loc[best_cat, 'Total_Revenue']:,.2f})
  ✅ Best Region       : {best_region}  (₹{rev_by_region[best_region]:,.2f})
  ✅ Best Sales Rep    : {best_rep}
  ✅ Avg Discount      : {avg_discount:.1f}%
  ✅ Avg Customer Rating: {df['Customer_Rating'].mean():.2f} / 5.0
""")

# ─────────────────────────────────────────────
# STEP 8: VISUALIZATIONS
# ─────────────────────────────────────────────
print("=" * 55)
print("  STEP 8: GENERATING CHARTS")
print("=" * 55)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sales Data Analysis — 2024', fontsize=16, fontweight='bold', y=1.01)

colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']

# Chart 1: Revenue by Category (Bar)
ax1 = axes[0, 0]
cats = rev_by_cat.index.tolist()
vals = rev_by_cat['Total_Revenue'].tolist()
bars = ax1.bar(cats, vals, color=colors[:len(cats)], edgecolor='white', linewidth=0.8)
ax1.set_title('Total Revenue by Category', fontweight='bold')
ax1.set_xlabel('Category')
ax1.set_ylabel('Revenue (₹)')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e5:.1f}L'))
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'₹{val/1e5:.1f}L', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.tick_params(axis='x', rotation=15)

# Chart 2: Revenue by Region (Pie)
ax2 = axes[0, 1]
ax2.pie(rev_by_region.values, labels=rev_by_region.index, autopct='%1.1f%%',
        colors=colors, startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax2.set_title('Revenue Distribution by Region', fontweight='bold')

# Chart 3: Monthly Revenue Trend (Line)
ax3 = axes[1, 0]
ax3.plot(monthly_rev['Month'], monthly_rev['Revenue'], marker='o', linewidth=2.5,
         color='#4C72B0', markersize=8, markerfacecolor='white', markeredgewidth=2)
ax3.fill_between(range(len(monthly_rev)), monthly_rev['Revenue'], alpha=0.15, color='#4C72B0')
ax3.set_title('Monthly Revenue Trend', fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Revenue (₹)')
ax3.set_xticks(range(len(monthly_rev)))
ax3.set_xticklabels(monthly_rev['Month'], rotation=30, ha='right', fontsize=8)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e5:.1f}L'))

# Chart 4: Sales Rep Performance (Horizontal Bar)
ax4 = axes[1, 1]
reps = rep_perf.index.tolist()
rev_vals = rep_perf['Total_Revenue'].tolist()
h_bars = ax4.barh(reps, rev_vals, color=colors[:len(reps)], edgecolor='white', height=0.5)
ax4.set_title('Sales Rep Performance (Revenue)', fontweight='bold')
ax4.set_xlabel('Revenue (₹)')
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e5:.1f}L'))
for bar, val in zip(h_bars, rev_vals):
    ax4.text(val + 2000, bar.get_y() + bar.get_height()/2,
             f'₹{val/1e5:.1f}L', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('sales_analysis_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Charts saved → sales_analysis_charts.png")

print("\n" + "=" * 55)
print("  ANALYSIS COMPLETE!")
print("=" * 55)
