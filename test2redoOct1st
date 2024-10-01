import pandas as pd
from google.colab import files

#upload the file
uploaded = files.upload()

#load the dataset
file_name = "SAF Dataset.xlsx"
df = pd.read_excel(file_name)

#display the first few rows
df.head()

# check the data types to identify numeric columns
df.dtypes

df['Plant capacity (kg/hr)'] = df['Plant capacity (kg/hr)'].astype(float)
# check data types again
df.dtypes

# check for missing data
missing_data = df.isnull().sum()

# display columns with missing data
missing_data[missing_data > 0]

# there is no missing data

# Identify numeric columns
numeric_columns = df.select_dtypes(include=['float']).columns

# Identify categorical columns (assuming they are of type 'object')
categorical_columns = df.select_dtypes(include=['object']).columns

# apply one hot encoded
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

df_encoded.head()

# Find duplicate columns
def get_duplicate_columns(df_encoded):
    duplicate_column_names = set()
    for i in range(df_encoded.shape[1]):
        col = df_encoded.iloc[:, i]
        for j in range(i + 1, df_encoded.shape[1]):
            other_col = df_encoded.iloc[:, j]
            if col.equals(other_col):
                duplicate_column_names.add(df_encoded.columns[j])
    return list(duplicate_column_names)
duplicate_columns = get_duplicate_columns(df_encoded) # Pass the DataFrame to the function
print("Duplicate columns:", duplicate_columns)

# Summary statistics for numerical columns
df.describe()

# Handling the Outliers
import matplotlib.pyplot as plt

# Create the box plot for the remaining columns
df.boxplot()
plt.title('Box Plots for Numerical Columns (Excluding Specified Columns)')
plt.xticks(rotation=45)
plt.show()

#Contains ouliers

# Handle outliers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Calculate the IQR for each numerical column
outlier_threshold = 0.5
q1 = df.select_dtypes(include=np.number).quantile(0.25) # Select numerical columns only
q3 = df.select_dtypes(include=np.number).quantile(0.75) # Select numerical columns only
iqr = q3 - q1
lower_bound = q1 - outlier_threshold * iqr
upper_bound = q3 + outlier_threshold * iqr

# Filter out outliers
df_without_outliers = df[(df[df.select_dtypes(include=np.number).columns] >= lower_bound) & (df[df.select_dtypes(include=np.number).columns] <= upper_bound)]

# Check for the data without outliers
df_without_outliers.boxplot()
plt.title('Box Plots for Numerical Columns (without outliers)')
plt.xticks(rotation=45)
plt.show()

# Outliers are handled by calculating IQR and then removing using lower and upper bound
# H, N, O, VM, Ash, Hem, Lig, Plant capacity outliers have been removed

import seaborn as sns
import matplotlib.pyplot as plt
# Compute correlation matrix for df_without_outliers
corr_matrix_outliers = df_without_outliers.corr()
print(corr_matrix_outliers)

# Plot correlation matrix heatmap for df_without_outliers
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_outliers, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap for df_without_outliers')
plt.show()

#There is strong correlation between N and Ash, O and FC,Plant Capacity and MSP, VM and Lig, S and Ash, C and O, Ash and Cel, O and S

if len(numeric_columns) > 1:
    sns.pairplot(df_without_outliers[numeric_columns].dropna())
    plt.show()
else:
    print("Insufficient numeric columns for pair plot.")

# Analyze the correlations between features and target variables

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select only numeric columns from the DataFrame
numeric_df = df_without_outliers.select_dtypes(include=[float])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Print the correlation matrix
print("Correlation Matrix:\n", corr_matrix)

# Identify pairs of highly correlated variables
threshold = 0.5
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("\nHighly Correlated Pairs (threshold = 0.5):", high_corr_pairs)

for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")

#Combining highly correlated variables
# Standardize the data
numeric_df = df_without_outliers.select_dtypes(include=[float]) # Use df_without_outliers instead of df
scaler = StandardScaler()

# Drop rows with NaN values before scaling
numeric_df = numeric_df.dropna() # Added line to drop rows with NaN values
scaled_data = scaler.fit_transform(numeric_df)

# Apply PCA to reduce multicollinearity
pca = PCA(n_components=len(numeric_df.columns))
pca.fit(scaled_data)

# Check explained variance to decide how many components to keep
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each principal component:", explained_variance)

cumulative_variance = explained_variance.cumsum()
n_components_to_keep = next(i for i, total_var in enumerate(cumulative_variance) if total_var > 0.95) + 1

# Transform data using the chosen number of components
df_pca = pca.transform(scaled_data)[:, :n_components_to_keep]

print(f"Reduced data using {n_components_to_keep} principal components.")

# There is strong positive correlation between O and FC, Lig and VM, MSP and Plant Capacity, Ash and N, indicating that when O increases, FC would also increase and similar to the rest
# There is strong negative correlation between O and C, O and S, Cel and Ash, indicating that when Cel increases, Ash would decrease and similar to the rest
