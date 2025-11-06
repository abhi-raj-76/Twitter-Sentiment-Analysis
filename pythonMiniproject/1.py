import pandas as pd

df = pd.read_csv('twitter_training.csv', encoding='utf-8', engine='python', on_bad_lines='skip')
#print("First 10 rows:")
#print(df.head(10).to_string(index=False))
#print("\nShape:", df.shape)
#print("\nData types:")
#print(df.dtypes)

# check for and handle missing values (use median/mode/fillmethods where appropriate).
print("\nMissing values:")
print(df.isnull().sum())    
df.fillna(method='ffill', inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())

#remove duplicates and correct obvious data entry errors.
initial_shape = df.shape
df.drop_duplicates(inplace=True)
final_shape = df.shape
print(f"\nDuplicates removed: {initial_shape[0] - final_shape[0]}") 

# encode categorical variables (label encoder or get dummies where needed). 
df = pd.get_dummies(df, drop_first=True)    
print("\nData after encoding categorical variables:")
print(df.head().to_string(index=False))

# create at least two derived features(feature engineering) using vectorized operations.
if 'text' in df.columns:
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    print("\nData after adding derived features:")
    print(df[['text_length', 'word_count']].head().to_string(index=False))
else:
    print("\nColumn 'text' not found for feature engineering.")
    df['text_length'] = 0
    df['word_count'] = 0
# detect and handle outliers in at least one numerical feature (IQR or Z-score method).
if 'text_length' in df.columns:
    Q1 = df['text_length'].quantile(0.25)
    Q3 = df['text_length'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['text_length'] < lower_bound) | (df['text_length'] > upper_bound)]
    print(f"\nOutliers detected in 'text_length': {outliers.shape[0]}")
    df = df[(df['text_length'] >= lower_bound) & (df['text_length'] <= upper_bound)]
    print(f"Data shape after removing outliers: {df.shape}")
else:
    print("\nColumn 'text_length' not found for outlier detection.")
print("\nFinal cleaned data:")
print(df.head(10).to_string(index=False))

#Normalize or scale numerical features using MinMaxScaler or StandardScaler.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("\nData after normalization:")
print(df.head().to_string(index=False))
# save cleaned data to new CSV file
df.to_csv('twitter_training_cleaned.csv', index=False)

#perform grouping/aggregation operations to answer a domain-specific question.
if 'text_length' in df.columns:
    avg_text_length = df['text_length'].mean()
    print(f"\nAverage text length after cleaning: {avg_text_length}")
else:  
    print("\nColumn 'text_length' not found for aggregation.")
    avg_text_length = 0
print("\nData cleaning and preprocessing completed. Cleaned data saved to 'twitter_training_cleaned.csv'.")

#Visualize results with at least two different types of plots (e.g., histograms, box plots, scatter plots).
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Histogram of Text Length')
plt.subplot(1, 2, 2)
sns.boxplot(x=df['text_length'])
plt.title('Box Plot of Text Length')
plt.tight_layout()
plt.show()

# Summarize findings,provide insights, and suggest next steps for modeling or deeper analysis.
print("\nSummary of Findings:")
print(f"- Initial data shape: {initial_shape}")
print(f"- Final data shape after cleaning: {df.shape}")
print(f"- Average text length after cleaning: {avg_text_length}")
print("\nNext Steps:")
print("- Explore relationships between features and target variable.")
print("- Consider advanced feature engineering techniques.")