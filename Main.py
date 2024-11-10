import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Combine all data into a single DataFrame
april_data = pd.read_csv('april_2024.csv', encoding='ISO-8859-1')
may_data = pd.read_csv('may_2024.csv', encoding='ISO-8859-1')
june_data = pd.read_csv('june_2024.csv', encoding='ISO-8859-1')
july_data = pd.read_csv('july_2024.csv', encoding='ISO-8859-1')
august_data = pd.read_csv('august_2024.csv', encoding='ISO-8859-1')
september_data = pd.read_csv('september_2024.csv', encoding='ISO-8859-1')
october_data = pd.read_csv('october_2024.csv', encoding='ISO-8859-1')

# Combine all data
combined_data = pd.concat([april_data, may_data, june_data, july_data, august_data, september_data, october_data], ignore_index=True)
combined_data = combined_data[~combined_data['Modifier'].str.contains('No', case=False, na=False)]
# Convert 'Sent Date' to datetime
combined_data['Sent Date'] = pd.to_datetime(combined_data['Sent Date'])

# Extract month from the date
combined_data['Month'] = combined_data['Sent Date'].dt.to_period('M')

# Count unique orders per month and menu selection
monthly_menu_counts = combined_data.groupby(['Month', 'Parent Menu Selection'])['Order ID'].nunique().unstack(fill_value=0)

# Plot stacked area chart
fig, ax = plt.subplots(figsize=(12, 6))
monthly_menu_counts.plot(kind='area', stacked=True, ax=ax)
plt.title('Monthly Trend of Menu Selections (April to October 2024)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend(title='Menu Selection', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels([label.get_text().split('-')[1] for label in ax.get_xticklabels()])
plt.tight_layout()
plt.show()

# Analyze popular modifiers for Mac and Cheese
mac_cheese_data = combined_data[combined_data['Parent Menu Selection'] == 'Mac and Cheese']
modifier_counts = mac_cheese_data.groupby('Modifier')['Order ID'].nunique().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
modifier_counts.plot(kind='bar', ax=ax)
plt.title('Top 10 Modifiers for Mac and Cheese (April to October 2024)', fontsize=15)
plt.xlabel('Modifier', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# Analyze popular toppings
toppings_data = combined_data[combined_data['Option Group Name'] == 'Choose Your Toppings']
topping_counts = toppings_data.groupby('Modifier')['Order ID'].nunique().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
topping_counts.plot(kind='bar', ax=ax)
plt.title('Popularity of Toppings (April to October 2024)', fontsize=15)
plt.xlabel('Topping', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# Analyze order patterns by hour of day
combined_data['Hour'] = combined_data['Sent Date'].dt.hour
hourly_orders = combined_data.groupby('Hour')['Order ID'].nunique()

fig, ax = plt.subplots(figsize=(12, 6))
hourly_orders.plot(kind='line', marker='o', ax=ax)
plt.title('Order Pattern by Hour of Day (April to October 2024)', fontsize=15)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(range(0, 24), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Monthly order volume
monthly_order_volume = combined_data.groupby('Month')['Order ID'].nunique()

fig, ax = plt.subplots(figsize=(12, 6))
monthly_order_volume.plot(kind='bar', ax=ax)
plt.title('Monthly Order Volume (April to October 2024)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
month_names = ['April', 'May', 'June', 'July', 'August', 'September', 'October']
ax.set_xticklabels(month_names, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print the order counts for verification
print(monthly_order_volume)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming the previous data loading and preprocessing steps are already done
print("Hours1",combined_data['Hour'])
# Feature engineering
combined_data['DayOfWeek'] = combined_data['Sent Date'].dt.dayofweek
combined_data['Month'] = combined_data['Sent Date'].dt.month
combined_data['Hour'] = combined_data['Sent Date'].dt.hour
print("Hours2",combined_data['Hour'])

# Aggregate data by day
daily_data = combined_data.groupby(['Sent Date', 'DayOfWeek', 'Month']).agg({
    'Order ID': 'nunique',
    'Parent Menu Selection': lambda x: x.value_counts().index[0],
    'Modifier': lambda x: x.value_counts().index[0]
}).reset_index()

daily_data.columns = ['Date', 'DayOfWeek', 'Month', 'OrderCount', 'TopMenuItem', 'TopModifier']

# Prepare features and target
X = daily_data[['DayOfWeek', 'Month']]
y = daily_data['OrderCount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['DayOfWeek', 'Month'])
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = model.named_steps['regressor'].feature_importances_
feature_names = ['DayOfWeek', 'Month']

for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance}")

# Predict future order volumes
future_dates = pd.date_range(start='2024-11-01', end='2024-12-31')
future_X = pd.DataFrame({
    'DayOfWeek': future_dates.dayofweek,
    'Month': future_dates.month
})

future_predictions = model.predict(future_X)

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_predictions, label='Predicted Orders')
plt.title('Predicted Daily Order Volume (November-December 2024)')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.legend()
plt.tight_layout()
plt.show()

# Analyze peak hours
hourly_orders = combined_data.groupby('Hour')['Order ID'].nunique()
print("combined",combined_data)
peak_hours = hourly_orders[hourly_orders > hourly_orders.mean() + hourly_orders.std()].index

print("Peak Hours:", peak_hours.tolist())

# Analyze popular menu items and modifiers
popular_items = combined_data['Parent Menu Selection'].value_counts().head(5)
popular_modifiers = combined_data['Modifier'].value_counts().head(10)

print("\nTop 5 Menu Items:")
print(popular_items)

print("\nTop 10 Modifiers:")
print(popular_modifiers)

# Calculate average items per order
items_per_order = combined_data.groupby('Order ID').size().mean()
print(f"\nAverage Items per Order: {items_per_order:.2f}")

# Provide insights and recommendations
print("\nInsights and Recommendations:")
print(f"1. Staffing: Increase staff by 30-50% during peak hours: {peak_hours.tolist()}")
print(f"2. Inventory: Ensure at least {int(popular_items.iloc[0] * 1.2/365)} servings of {popular_items.index[0]} are prepared daily")
print(f"3. Modifiers: Stock up on {popular_modifiers.index[0]} and {popular_modifiers.index[1]}, with at least {int(popular_modifiers.iloc[0] * 1.2)} servings each")
print(f"4. Order Efficiency: Prepare for an average of {items_per_order:.2f} items per order")
print("5. Menu Optimization: Consider creating combo meals based on popular item and modifier combinations")
print("6. Demand Forecasting: Use the predictive model to adjust staffing and inventory for upcoming months")