import pandas as pd

#load the dataset
file_name = "Test2_dataset.xlsx"
df = pd.read_excel(file_name)

#display the first few rows
df.head()

# print the dataset's columns
print(df.columns)

# check the data types to identify numeric columns
df.dtypes

# check for missing data
missing_data = df.isnull().sum()

# display columns with missing data
missing_data[missing_data > 0]

# impute missing data with the mean
df['Ni Dispersion'].fillna(df['Ni Dispersion'].mean(), inplace=True)
df['CH4 Conversion'].fillna(df['CH4 Conversion'].mean(), inplace=True)
df['CO2 Conversion'].fillna(df['CO2 Conversion'].mean(), inplace=True)
df['Syngas_Ratio'].fillna(df['Syngas_Ratio'].mean(), inplace=True)

# Identify numeric columns
numeric_columns = df.select_dtypes(include=['float', 'int']).columns

# Identify categorical columns (assuming they are of type 'object')
categorical_columns = df.select_dtypes(include=['object']).columns

# Change int to float
df['Reaction Temperature'] = df['Reaction Temperature'].astype(float)
df['GHSV'] = df['GHSV'].astype(float)
print(df.dtypes)

# Encoding categorical variables

# Apply one-hot-encoding
df = pd.get_dummies(df, columns=['Catalyst'])

# Handling the Outliers
import matplotlib.pyplot as plt

# Create the box plot for the remaining columns
df.boxplot()
plt.title('Box Plots for Numerical Columns (Excluding Specified Columns)')
plt.xticks(rotation=45)
plt.show()

#Contains ouliers

import numpy as np 
# Calculate the IQR for each numerical column
outlier_threshold = 1.5
q1 = df.select_dtypes(include=np.number).quantile(0.25) # Select numerical columns only
q3 = df.select_dtypes(include=np.number).quantile(0.75) # Select numerical columns only
iqr = q3 - q1
lower_bound = q1 - outlier_threshold * iqr
upper_bound = q3 + outlier_threshold * iqr

# Filter out outliers
df_without_outliers = df[(df[df.select_dtypes(include=np.number).columns] >= lower_bound) & (df[df.select_dtypes(include=np.number).columns] <= upper_bound)] # Use df instead of filtered_df

# Create the box plot without outliers
df_without_outliers.boxplot()
plt.title('Box Plots for Numerical Columns (without outliers)')
plt.xticks(rotation=45)
plt.show()

# Outliers are handled by calculating IQR and then removing using lower and upper bound

# Summarize key characteristics of the data without outliers
summary_stats = df_without_outliers.describe().T
summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
summary_stats = summary_stats[['mean', '50%', 'std', 'min', 'max', 'IQR']]
summary_stats.columns = ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum', 'IQR']

print(summary_stats)

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

if len(numeric_columns) > 1:
    sns.pairplot(df_without_outliers[numeric_columns].dropna())
    plt.show()
else:
    print("Insufficient numeric columns for pair plot.")
    

# Analyze the correlations between features and target variables 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select only numeric columns from the DataFrame
numeric_df = df_without_outliers.select_dtypes(include=[float, int])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Print the correlation matrix
print("Correlation Matrix:\n", corr_matrix)

# Identify pairs of highly correlated variables
threshold = 0.9
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("\nHighly Correlated Pairs (threshold = 0.9):")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")

#Combining highly correlated variables
# Standardize the data
numeric_df = df_without_outliers.select_dtypes(include=[float, int]) # Use df_without_outliers instead of df
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

