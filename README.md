# Twitter Sentiment Analysis

A machine learning project that performs sentiment analysis on Twitter data using Random Forest and Support Vector Machine (SVM) classifiers.

## 📋 Overview

This project implements a complete machine learning pipeline for sentiment analysis, including data preprocessing, feature engineering, visualization, and model training with performance evaluation.

## 🚀 Features

- **Data Preprocessing**: Handles missing values, removes duplicates, and cleans the dataset
- **Feature Engineering**: Creates derived features like text length and word count
- **Outlier Detection**: Uses IQR method to identify and remove outliers
- **Feature Scaling**: Normalizes numerical features using MinMaxScaler
- **Visualization**: Generates comprehensive plots including histograms, boxplots, heatmaps, and pairplots
- **Model Training**: Implements Random Forest and SVM classifiers
- **Performance Metrics**: Provides accuracy scores, classification reports, and confusion matrices

## 📦 Requirements
```bash
pandas
matplotlib
seaborn
scikit-learn
```

Install dependencies using:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## 📁 Dataset

The project expects a Twitter training dataset in CSV format with the following structure:
- **File**: `twitter_training.csv`
- **Format**: Tab-separated values with latin-1 encoding
- **Columns**: `id`, `entity`, `sentiment`, `text`

## 🔧 Usage

Run the script:
```bash
python twitter_sentiment_analysis.py
```

## 📊 Pipeline Steps

1. **Load Dataset**: Reads the Twitter training data
2. **Handle Missing Values**: Fills missing text entries and forward-fills other columns
3. **Remove Duplicates**: Eliminates duplicate entries
4. **Encode Labels**: Converts sentiment labels to numerical values
5. **Feature Engineering**: Creates text_length and word_count features
6. **Outlier Detection**: Removes outliers using IQR method
7. **Feature Scaling**: Normalizes features to 0-1 range
8. **Save Cleaned Data**: Exports processed data to CSV
9. **Visualization**: Generates multiple plots for data analysis
10. **Train/Test Split**: Prepares data for modeling (80/20 split)
11. **Random Forest Training**: Trains and evaluates RF classifier
12. **SVM Training**: Trains and evaluates SVM classifier

## 📈 Output Files

### Data
- `twitter_training_cleaned_final.csv` - Cleaned and processed dataset

### Visualizations
- `step9_histogram_text_length.png` - Distribution of text lengths
- `step9_boxplot_text_length.png` - Boxplot showing text length outliers
- `step9_heatmap_correlation.png` - Feature correlation heatmap
- `step9_pairplot_features.png` - Pairwise feature relationships
- `step11_confusion_matrix_rf.png` - Random Forest confusion matrix
- `step12_confusion_matrix_svm.png` - SVM confusion matrix

## 🎯 Model Performance

The script outputs:
- Accuracy scores for both models
- Detailed classification reports with precision, recall, and F1-scores
- Confusion matrices visualizing prediction performance

## 🛠️ Model Configuration

### Random Forest
- **Estimators**: 150 trees
- **Random State**: 42

### SVM
- **Kernel**: RBF
- **C Parameter**: 1.0
- **Gamma**: scale
- **Random State**: 42

## 📝 Notes

- The script uses stratified train-test split to maintain class distribution
- Small classes (< 2 samples) are automatically removed before training
- All visualizations use seaborn's whitegrid style for consistent aesthetics

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: Make sure your dataset is properly formatted and placed in the same directory as the script before running.
