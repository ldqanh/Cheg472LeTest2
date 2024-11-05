Steps Included in the Quiz

1. Data Import and Initial Inspection

The script begins by uploading an Excel file named "SAF Dataset.xlsx" and loading it into a DataFrame.
Displays the first few rows of the data and checks the data types to identify numerical columns.

2. Data Type Conversion and Missing Data Check

Converts the column Plant capacity (kg/hr) to a float to ensure consistent numerical data types.
Checks for any missing data, though none is found in this dataset.

3. Categorical Encoding and Duplicate Column Identification

Identifies categorical columns and applies one-hot encoding to create binary columns for each category, excluding the first category to avoid multicollinearity.
A function is included to identify duplicate columns, ensuring data redundancy is minimized.

4. Summary Statistics and Outlier Detection

Generates summary statistics for numerical columns.
Outliers are detected and visualized using box plots.

5. Outlier Handling Using IQR

Outliers are handled using the Interquartile Range (IQR) method, filtering out rows with values outside specified thresholds.
Visualizes the data after outlier removal to confirm the absence of outliers in key numerical columns.

6. Correlation Analysis

Computes and visualizes the correlation matrix as a heatmap to explore relationships between numerical features.
Pairs of highly correlated variables are identified for further analysis, which helps in understanding feature interactions within the dataset.

7. Dimensionality Reduction Using PCA

Standardizes numerical features and applies PCA to reduce dimensionality and multicollinearity.
Based on explained variance, selects the minimum number of principal components that together account for over 95% of the total variance.

8. Key Findings

Strong positive and negative correlations are observed among several features. For instance:
Positive correlations (e.g., Oxygen and FC, Lignin and VM) indicate mutual increases, while negative correlations (e.g., Cellulose and Ash) show opposing trends.
Dimensionality reduction using PCA is successful, reducing the data to essential principal components for further analysis.

Libraries Used
- Pandas: For data handling and manipulation.
- Matplotlib and Seaborn: For data visualization, including box plots and heatmaps.
- NumPy: For numerical operations and handling outlier thresholds.
- Scikit-Learn: For scaling data and performing PCA for dimensionality reduction.
