from collections import Counter
import streamlit as st
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
combined_data = pd.concat([april_data, may_data, june_data, july_data, august_data, september_data, october_data],
                          ignore_index=True)
combined_data = combined_data[~combined_data['Modifier'].str.contains('No', case=False, na=False)]
# Convert 'Sent Date' to datetime
combined_data['Sent Date'] = pd.to_datetime(combined_data['Sent Date'])

# Extract month from the date
combined_data['Month'] = combined_data['Sent Date'].dt.to_period('M')

# Count unique orders per month and menu selection
monthly_menu_counts = combined_data.groupby(['Month', 'Parent Menu Selection'])['Order ID'].nunique().unstack(
    fill_value=0)

# Plot stacked area chart
title = '<p style="font-family:sans-serif; color:Green; font-size: 50px;">Roni\'s Data Analysis</p>'
st.markdown(title, unsafe_allow_html=True)
image1_title= '<p style="font-family:sans-serif; color:White; font-size: 35px;">Monthly Menu Selections</p>'
st.markdown(image1_title, unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
monthly_menu_counts.plot(kind='area', stacked=True, ax=ax)
plt.title('Monthly Trend of Menu Selections (April to October 2024)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend(title='Menu Selection', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels([label.get_text().split('-')[1] for label in ax.get_xticklabels()])
plt.tight_layout()
st.pyplot(plt)
plt.show()

# Analyze popular modifiers for Mac and Cheese
mac_cheese_data = combined_data[combined_data['Parent Menu Selection'] == 'Mac and Cheese']
modifier_counts = mac_cheese_data.groupby('Modifier')['Order ID'].nunique().sort_values(ascending=False).head(10)

image2_title= '<p style="font-family:sans-serif; color:White; font-size: 35px;">Top 10 Modifiers</p>'
st.markdown(image2_title, unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
modifier_counts.plot(kind='bar', ax=ax)
plt.title('Top 10 Modifiers for Mac and Cheese (April to October 2024)', fontsize=15)
plt.xlabel('Modifier', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
st.pyplot(plt)
plt.show()

# Analyze popular toppings
toppings_data = combined_data[combined_data['Option Group Name'] == 'Choose Your Toppings']
topping_counts = toppings_data.groupby('Modifier')['Order ID'].nunique().sort_values(ascending=False)

image3_title= '<p style="font-family:sans-serif; color:White; font-size: 35px;">Popular Toppings</p>'
st.markdown(image3_title, unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
topping_counts.plot(kind='bar', ax=ax)
plt.title('Popularity of Toppings (April to October 2024)', fontsize=15)
plt.xlabel('Topping', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
st.pyplot(plt)
plt.show()

# Analyze order patterns by hour of day
combined_data['Hour'] = combined_data['Sent Date'].dt.hour
hourly_orders = combined_data.groupby('Hour')['Order ID'].nunique()

image4_title= '<p style="font-family:sans-serif; color:White; font-size: 35px;">Orders By Hour</p>'
st.markdown(image4_title, unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
hourly_orders.plot(kind='line', marker='o', ax=ax)
plt.title('Order Pattern by Hour of Day (April to October 2024)', fontsize=15)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(range(0, 24), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(plt)
plt.show()

# Monthly order volume
monthly_order_volume = combined_data.groupby('Month')['Order ID'].nunique()

image5_title= '<p style="font-family:sans-serif; color:White; font-size: 35px;">Monthly Order Number</p>'
st.markdown(image5_title, unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
monthly_order_volume.plot(kind='bar', ax=ax)
plt.title('Monthly Order Number (April to October 2024)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
month_names = ['April', 'May', 'June', 'July', 'August', 'September', 'October']
ax.set_xticklabels(month_names, rotation=45, ha='right')
plt.tight_layout()
st.pyplot(plt)
plt.show()

# Print the order counts for verification
print(monthly_order_volume)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta


# Load and preprocess data
csv_files = ['april_2024.csv', 'may_2024.csv', 'june_2024.csv', 'july_2024.csv',
             'august_2024.csv', 'september_2024.csv', 'october_2024.csv']

combined_data = pd.concat([pd.read_csv(file, encoding='ISO-8859-1') for file in csv_files], ignore_index=True)
combined_data = combined_data[~combined_data['Modifier'].str.contains('No', case=False, na=False)]

# Convert 'Sent Date' to datetime
combined_data['Sent Date'] = pd.to_datetime(combined_data['Sent Date'])

# Group by date and get the highest order number for each day
daily_orders = combined_data.groupby(combined_data['Sent Date'].dt.date)['Order #'].max().reset_index()
daily_orders.columns = ['Date', 'OrderCount']

# Add features
daily_orders['Date'] = pd.to_datetime(daily_orders['Date'])
daily_orders['DayOfWeek'] = daily_orders['Date'].dt.dayofweek
daily_orders['Month'] = daily_orders['Date'].dt.month
daily_orders['DayOfMonth'] = daily_orders['Date'].dt.day

# Add school session feature
def is_school_session(date):
    month = date.month
    day = date.day
    if (month == 8 and day >= 15) or (month > 8 and month < 12) or (month == 12 and day <= 15):
        return 1  # Fall semester
    elif (month == 1 and day >= 15) or (month > 1 and month < 5) or (month == 5 and day <= 15):
        return 1  # Spring semester
    else:
        return 0  # Break

daily_orders['SchoolSession'] = daily_orders['Date'].apply(is_school_session)

# Prepare features and target
X = daily_orders[['DayOfWeek', 'Month', 'DayOfMonth', 'SchoolSession']]
y = daily_orders['OrderCount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = model.feature_importances_
feature_names = X.columns

for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance}")

# Streamlit app layout
image6_title= '<p style="font-family:sans-serif; color:White; font-size: 35px;"> Order Volume Prediction</p>'
st.markdown(image6_title, unsafe_allow_html=True)

# Date selector for specific predictions
selected_date = st.date_input('Select a date for prediction', value=pd.to_datetime('2025-01-01'))

# Prepare input for prediction based on selected date
selected_X = pd.DataFrame({
    'DayOfWeek': [selected_date.weekday()],
    'Month': [selected_date.month],
    'DayOfMonth': [selected_date.day],
    'SchoolSession': [is_school_session(selected_date)]
})

# Make prediction for the selected date
prediction = model.predict(selected_X)[0]
prediction_clipped = np.clip(prediction, 0, 750)  # Clip to max of 750 orders

st.write(f"Predicted order volume for {selected_date}: {prediction_clipped:.0f}")

# Predict future order volumes for a full year (November 2024 - October 2025)
future_dates = pd.date_range(start='2024-11-01', end='2025-10-31')
future_X = pd.DataFrame({
    'DayOfWeek': future_dates.dayofweek,
    'Month': future_dates.month,
    'DayOfMonth': future_dates.day,
    'SchoolSession': [is_school_session(date) for date in future_dates]
})

future_predictions = model.predict(future_X)

# Clip predictions to avoid unrealistic values (e.g., over a certain threshold)
future_predictions = np.clip(future_predictions, 0, 750)  # Set max to 750 orders

# Visualize predictions for the full year
plt.figure(figsize=(16, 8))
plt.plot(future_dates, future_predictions, label='Predicted Orders')
plt.title('Predicted Daily Order Volume (November 2024 - October 2025)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(plt)
plt.show()

# Print some statistics about the predictions
st.write("\nPrediction Statistics:")
st.write(f"Average predicted daily orders: {future_predictions.mean():.2f}")
st.write(f"Minimum predicted daily orders: {future_predictions.min():.2f}")
st.write(f"Maximum predicted daily orders: {future_predictions.max():.2f}")

# Print monthly averages for better insights
monthly_averages = pd.DataFrame({
    'Month': future_dates.month,
    'Predictions': future_predictions
}).groupby('Month')['Predictions'].mean()

st.write("\nMonthly average predictions:")
st.write(monthly_averages)