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

combined_data = pd.concat([april_data, may_data, june_data, july_data, august_data, september_data], ignore_index=True)

# Convert 'Sent Date' to datetime
combined_data['Sent Date'] = pd.to_datetime(combined_data['Sent Date'])

# Extract month from the date
combined_data['Month'] = combined_data['Sent Date'].dt.to_period('M')

# Group by month and parent menu selection
monthly_menu_counts = combined_data.groupby(['Month', 'Parent Menu Selection']).size().unstack(fill_value=0)

# Plot stacked area chart
fig, ax = plt.subplots(figsize=(10, 5))
monthly_menu_counts.plot(kind='area', stacked=True, ax=ax)
plt.title('Monthly Trend of Menu Selections (April to September 2024)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend(title='Menu Selection', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Analyze popular modifiers for Mac and Cheese
mac_cheese_data = combined_data[combined_data['Parent Menu Selection'] == 'Mac and Cheese']
modifier_counts = mac_cheese_data['Modifier'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
modifier_counts.plot(kind='bar', ax=ax)
plt.title('Top 10 Modifiers for Mac and Cheese (April to September 2024)', fontsize=15)
plt.xlabel('Modifier', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# Analyze popular toppings
toppings_data = combined_data[combined_data['Option Group Name'] == 'Choose Your Toppings']
topping_counts = toppings_data['Modifier'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
topping_counts.plot(kind='bar', ax=ax)
plt.title('Popularity of Toppings (April to September 2024)', fontsize=15)
plt.xlabel('Topping', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# Analyze order patterns by hour of day
combined_data['Hour'] = combined_data['Sent Date'].dt.hour
hourly_orders = combined_data.groupby('Hour').size()

fig, ax = plt.subplots(figsize=(10, 5))
hourly_orders.plot(kind='line', marker='o', ax=ax)
plt.title('Order Pattern by Hour of Day (April to September 2024)', fontsize=15)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(range(0, 24), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Monthly order volume
monthly_order_volume = combined_data.groupby('Month').size()

fig, ax = plt.subplots(figsize=(10, 5))
monthly_order_volume.plot(kind='bar', ax=ax)
plt.title('Monthly Order Volume (April to September 2024)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()