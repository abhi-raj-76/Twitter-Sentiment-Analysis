import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load properly
df = pd.read_csv(
    'twitter_training.csv',
    encoding='latin-1',
    header=None,
    names=['sentiment', 'id', 'date', 'query', 'user', 'text'],
    engine='python',
    on_bad_lines='skip',
    quoting=3  # ignore quotes in text
)

print("✅ Data loaded successfully")
print(df.head(3))
print(f"Shape: {df.shape}")

# Step 2: Handle missing values
print("\nMissing values:")
print(df.isnull().sum())
df.fillna(method='ffill', inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())

# Step 3: Remove duplicates
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {initial_shape[0] - df.shape[0]}")

# Step 4: Feature engineering
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# Step 5: Outlier detection using IQR
Q1 = df['text_length'].quantile(0.25)
Q3 = df['text_length'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['text_length'] >= lower) & (df['text_length'] <= upper)]

# Step 6: Scale numeric features
scaler = MinMaxScaler()
df[['text_length', 'word_count']] = scaler.fit_transform(df[['text_length', 'word_count']])

# Step 7: Save cleaned data
df.to_csv('twitter_training_cleaned.csv', index=False)
print("\nCleaned data saved to 'twitter_training_cleaned.csv'")

# Step 8: Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Histogram of Text Length')
plt.subplot(1, 2, 2)
sns.boxplot(x=df['text_length'])
plt.title('Box Plot of Text Length')
plt.tight_layout()
plt.show()

# Step 9: Summary
print("\nSummary of Findings:")
print(f"- Final data shape: {df.shape}")
print(f"- Average text length: {df['text_length'].mean():.4f}")
