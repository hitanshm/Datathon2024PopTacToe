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