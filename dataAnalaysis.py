# Attempt to load the files with 'ISO-8859-1' encoding to handle any special characters
import pandas as pd
file_paths={'april': 'april_2024.csv', 'may': 'may_2024.csv', 'june': 'july_2024.csv','july': 'july_2024.csv', 'august': 'august_2024.csv',}
dataframes = {}
for month, path in file_paths.items():
    try:
        dataframes[month] = pd.read_csv(path, encoding='ISO-8859-1')
    except Exception as e:
        dataframes[month] = f"Error: {e}"

# Display the first few rows of each file to understand the structure
dataframes_preview = {month: df.head() if isinstance(df, pd.DataFrame) else df for month, df in dataframes.items()}
print(dataframes_preview)
