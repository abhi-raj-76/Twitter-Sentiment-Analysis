"""
Twitter Sentiment Analysis using Random Forest and SVM
"""

# =========================================================
# Imports
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# =========================================================
# STEP 1: Load dataset
# =========================================================
print("\nLoading dataset...")

df = pd.read_csv(
    'twitter_training.csv',
    encoding='latin-1',
    sep='\t',
    header=None,
    names=['id', 'entity', 'sentiment', 'text'],
    engine='python',
    on_bad_lines='skip',
    quoting=3
)

print(f"Data loaded successfully! Shape: {df.shape}")
print(df.head(5))

# =========================================================
# STEP 2: Handle missing values
# =========================================================
print("\nChecking for missing values...")
print(df.isnull().sum())

df['text'] = df['text'].fillna('')
df.ffill(inplace=True)

print("Missing values handled.")
print(df.isnull().sum())

# =========================================================
# STEP 3: Remove duplicates
# =========================================================
print("\nRemoving duplicates...")
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {initial_shape[0] - df.shape[0]}")

# =========================================================
# STEP 4: Encode categorical sentiment labels
# =========================================================
print("\nEncoding sentiment labels...")
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

# =========================================================
# STEP 5: Feature Engineering
# =========================================================
print("\nCreating derived text features...")
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

print("Derived features added:")
print(df[['sentiment', 'text_length', 'word_count']].head())

# =========================================================
# STEP 6: Outlier Detection (IQR Method)
# =========================================================
print("\nDetecting and removing outliers in text_length...")

Q1 = df['text_length'].quantile(0.25)
Q3 = df['text_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['text_length'] >= lower_bound) & (df['text_length'] <= upper_bound)]
print(f"Data shape after removing outliers: {df.shape}")

# =========================================================
# STEP 7: Scaling numerical features
# =========================================================
print("\nScaling numerical features...")
scaler = MinMaxScaler()
df[['text_length', 'word_count']] = scaler.fit_transform(df[['text_length', 'word_count']])
print("Scaling completed.")

# =========================================================
# STEP 8: Save cleaned data
# =========================================================
print("\nSaving cleaned data to CSV...")
df.to_csv('twitter_training_cleaned_final.csv', index=False)
print("Cleaned data saved as 'twitter_training_cleaned_final.csv'")

# =========================================================
# STEP 9: Visualization
# =========================================================
print("\nVisualizing data distributions...")

# Histogram of text_length
plt.figure()
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Histogram of Text Length')
plt.savefig('step9_histogram_text_length.png')
plt.close()

# Boxplot of text_length
plt.figure()
sns.boxplot(x=df['text_length'])
plt.title('Boxplot of Text Length')
plt.savefig('step9_boxplot_text_length.png')
plt.close()

# Heatmap for correlation
plt.figure()
sns.heatmap(df[['text_length', 'word_count', 'sentiment_encoded']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('step9_heatmap_correlation.png')
plt.close()

# Pairplot for feature relationships
sns.pairplot(df[['text_length', 'word_count', 'sentiment_encoded']], diag_kind='kde')
plt.savefig('step9_pairplot_features.png')
plt.close()

# =========================================================
# STEP 10: Prepare for Modeling
# =========================================================
print("\nPreparing data for machine learning...")

X = df[['text_length', 'word_count']]
y = df['sentiment_encoded']

# Check for small classes
if y.value_counts().min() < 2:
    print("Some classes have fewer than 2 samples. Removing them for training...")
    valid_classes = y.value_counts()[y.value_counts() >= 2].index
    df = df[df['sentiment_encoded'].isin(valid_classes)]
    X = df[['text_length', 'word_count']]
    y = df['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/Test split completed.")
print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

# =========================================================
# STEP 11: Random Forest Classifier
# =========================================================
print("\nTraining Random Forest Classifier...")

rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.3f}")

# Dynamically detect valid class names
unique_labels = sorted(y_test.unique())
unique_names = le.inverse_transform(unique_labels)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred, labels=unique_labels, target_names=unique_names))

rf_cm = confusion_matrix(y_test, rf_pred, labels=unique_labels)
plt.figure()
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=unique_names, yticklabels=unique_names)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('step11_confusion_matrix_rf.png')
plt.close()

# =========================================================
# STEP 12: Support Vector Machine (SVM)
# =========================================================
print("\nTraining Support Vector Machine (SVM)...")

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_acc:.3f}")
print("\nClassification Report (SVM):")
print(classification_report(y_test, svm_pred, labels=unique_labels, target_names=unique_names))

svm_cm = confusion_matrix(y_test, svm_pred, labels=unique_labels)
plt.figure()
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_names, yticklabels=unique_names)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('step12_confusion_matrix_svm.png')
plt.close()
