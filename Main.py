import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Combine all data into a single DataFrame
combined_data = pd.concat([april_data, june_data, july_data, august_data], ignore_index=True)

# Convert 'Sent Date' to datetime
combined_data['Sent Date'] = pd.to_datetime(combined_data['Sent Date'])

# Extract month from the date
combined_data['Month'] = combined_data['Sent Date'].dt.to_period('M')

# Group by month and parent menu selection
monthly_menu_counts = combined_data.groupby(['Month', 'Parent Menu Selection']).size().unstack(fill_value=0)

# Plot stacked area chart
plt.figure(figsize=(12, 6))
monthly_menu_counts.plot(kind='area', stacked=True)
plt.title('Monthly Trend of Menu Selections')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.legend(title='Menu Selection', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Analyze popular modifiers for Mac and Cheese
mac_cheese_data = combined_data[combined_data['Parent Menu Selection'] == 'Mac and Cheese']
modifier_counts = mac_cheese_data['Modifier'].value_counts().head(10)

plt.figure(figsize=(10, 6))
modifier_counts.plot(kind='bar')
plt.title('Top 10 Modifiers for Mac and Cheese')
plt.xlabel('Modifier')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Analyze popular toppings
toppings_data = combined_data[combined_data['Option Group Name'] == 'Choose Your Toppings']
topping_counts = toppings_data['Modifier'].value_counts()

plt.figure(figsize=(10, 6))
topping_counts.plot(kind='bar')
plt.title('Popularity of Toppings')
plt.xlabel('Topping')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Analyze order patterns by hour of day
combined_data['Hour'] = combined_data['Sent Date'].dt.hour
hourly_orders = combined_data.groupby('Hour').size()

plt.figure(figsize=(10, 6))
hourly_orders.plot(kind='line', marker='o')
plt.title('Order Pattern by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Orders')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()