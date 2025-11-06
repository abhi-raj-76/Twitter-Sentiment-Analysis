import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load dataset 
print("\n Loading dataset...")

df = pd.read_csv(
    'twitter_training.csv',
    encoding='latin-1',
    sep='\t',  # Tab-separated data
    header=None,
    names=['id', 'entity', 'sentiment', 'text'],
    engine='python',
    on_bad_lines='skip',
    quoting=3
)

print(f" Data loaded successfully! Shape: {df.shape}")
print(df.head(5))

# ---------------------------------------------------------
# STEP 2: Handle missing values
# ---------------------------------------------------------
print("\n Checking for missing values...")
print(df.isnull().sum())

# Fill text NaNs with empty strings
df['text'] = df['text'].fillna('')
# Forward fill for other columns
df.ffill(inplace=True)

print("\n Missing values handled.")
print(df.isnull().sum())

# ---------------------------------------------------------
# STEP 3: Remove duplicates
# ---------------------------------------------------------
print("\n Removing duplicates...")
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f" Duplicates removed: {initial_shape[0] - df.shape[0]}")

# ---------------------------------------------------------
# STEP 4: Encode categorical sentiment labels
# ---------------------------------------------------------
print("\n Encoding sentiment labels...")
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

# ---------------------------------------------------------
# STEP 5: Feature Engineering
# ---------------------------------------------------------
print("\n Creating derived text features...")
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

print("\n Derived features added:")
print(df[['sentiment', 'text_length', 'word_count']].head())

# ---------------------------------------------------------
# STEP 6: Outlier Detection (IQR Method)
# ---------------------------------------------------------
print("\n Detecting and removing outliers in text_length...")

Q1 = df['text_length'].quantile(0.25)
Q3 = df['text_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['text_length'] < lower_bound) | (df['text_length'] > upper_bound)]
print(f"Outliers detected: {outliers.shape[0]}")

df = df[(df['text_length'] >= lower_bound) & (df['text_length'] <= upper_bound)]
print(f" Data shape after removing outliers: {df.shape}")

# ---------------------------------------------------------
# STEP 7: Scaling numerical features
# ---------------------------------------------------------
print("\n Scaling numerical features...")

scaler = MinMaxScaler()
df[['text_length', 'word_count']] = scaler.fit_transform(df[['text_length', 'word_count']])

print(" Scaling completed.")
print(df[['text_length', 'word_count']].head())

# ---------------------------------------------------------
# STEP 8: Save cleaned data
# ---------------------------------------------------------
print("\n Saving cleaned data to CSV...")
df.to_csv('twitter_training_cleaned.csv', index=False)
print(" Cleaned data saved as 'twitter_training_cleaned.csv'")

# ---------------------------------------------------------
# STEP 9: Visualization
# ---------------------------------------------------------
print("\n Visualizing distributions...")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Histogram of Text Length')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['text_length'])
plt.title('Box Plot of Text Length')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# STEP 10: Summary & Insights
# ---------------------------------------------------------
avg_length = df['text_length'].mean()
avg_words = df['word_count'].mean()
print("\n Summary of Findings:")
print(f"- Final data shape: {df.shape}")
print(f"- Average text length: {avg_length:.3f}")
print(f"- Average word count: {avg_words:.3f}")

# ---------------------------------------------------------
# STEP 11: Train/Test Split for ML
# ---------------------------------------------------------
print("\n Preparing train/test split for modeling...")

X = df[['text', 'text_length', 'word_count']]
y = df['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(" Train/test split completed.")
print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}")

print("\n Data cleaning, feature engineering, and preparation completed successfully!")
