import pandas as pd
import os

# 1. Load the dataset
file_path = os.path.join(os.path.dirname(__file__), "memory_dataset_labeled (1).csv")
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# 2. Display column names
print("\nColumns:")
print(df.columns.tolist())

# 3. Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# 4. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 6. Check target values
print("\nTarget distribution:")
print(df["pressure_next_5s"].value_counts())

# 7. Sort data by experiment and timestamp
df = df.sort_values(["experiment", "timestamp"])

# 8. Remove rows with missing values
df = df.dropna()

# 9. Remove duplicate rows
df = df.drop_duplicates()

# 10. Save cleaned dataset
output_file = "memory_cleaned.csv"
df.to_csv(output_file, index=False)

print("\nCleaned shape:", df.shape)
print("Saved as:", output_file)

print("\nCleaning complete!")