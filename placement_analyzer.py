# ============================================
# AI-Powered Placement Intelligence System
# Complete Data Preprocessing + EDA + ML Code
# ============================================

# ---------- IMPORT LIBRARIES ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------- LOAD DATASET ----------

# Replace with your dataset filename
df = pd.read_csv("Placement_Data_Full_Class.csv")

# ---------- BASIC DATASET INFO ----------

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ============================================
# DATA PREPROCESSING
# ============================================

# ---------- HANDLE MISSING VALUES ----------

# Salary has null values for students not placed
# Fill with 0

df['salary'] = df['salary'].fillna(0)

# ---------- REMOVE DUPLICATES ----------

print("\nDuplicates Before Removal:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates After Removal:", df.duplicated().sum())

# ---------- ENCODE CATEGORICAL COLUMNS ----------

# Gender
df['gender'] = df['gender'].map({
    'M': 1,
    'F': 0
})

# SSC Board
df['ssc_b'] = df['ssc_b'].map({
    'Central': 1,
    'Others': 0
})

# HSC Board
df['hsc_b'] = df['hsc_b'].map({
    'Central': 1,
    'Others': 0
})

# Work Experience
df['workex'] = df['workex'].map({
    'Yes': 1,
    'No': 0
})

# Placement Status
df['status'] = df['status'].map({
    'Placed': 1,
    'Not Placed': 0
})

# ---------- ONE HOT ENCODING ----------

# For columns with multiple categories

df = pd.get_dummies(
    df,
    columns=[
        'hsc_s',
        'degree_t',
        'specialisation'
    ],
    drop_first=True
)

bool_columns = df.select_dtypes(include='bool').columns

df[bool_columns] = df[bool_columns].astype(int)

# ---------- DROP UNNECESSARY COLUMNS ----------

# sl_no is just serial number

df = df.drop('sl_no', axis=1)

# ---------- CHECK CLEANED DATA ----------

print("\n===== CLEANED DATA =====")
print(df.head())

print("\n===== CLEANED DATA INFO =====")
print(df.info())

# ---------- SAVE CLEANED DATASET ----------

df.to_csv("cleaned_placement_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ============================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

sns.set_style("whitegrid")

# ---------- PLACEMENT DISTRIBUTION ----------

plt.figure(figsize=(6,5))

sns.countplot(
    x='status',
    data=df
)

plt.title("Placement Distribution")
plt.xticks([0,1], ['Not Placed', 'Placed'])

plt.show()

# ---------- CGPA / MBA MARKS VS PLACEMENT ----------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='status',
    y='mba_p',
    data=df
)

plt.title("MBA Percentage vs Placement Status")
plt.xticks([0,1], ['Not Placed', 'Placed'])

plt.show()

# ---------- EMPLOYABILITY TEST SCORE ----------

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x='etest_p',
    hue='status',
    kde=True
)

plt.title("Employability Test Score Distribution")

plt.show()

# ---------- WORK EXPERIENCE IMPACT ----------

plt.figure(figsize=(6,5))

sns.countplot(
    x='workex',
    hue='status',
    data=df
)

plt.title("Work Experience vs Placement")

plt.show()

# ---------- SALARY DISTRIBUTION ----------

plt.figure(figsize=(8,5))

sns.histplot(
    df[df['salary'] > 0]['salary'],
    bins=20,
    kde=True
)

plt.title("Salary Distribution")

plt.show()

# ---------- CORRELATION HEATMAP ----------

plt.figure(figsize=(14,10))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================
# FEATURE SELECTION
# ============================================

# Target variable
y = df['status']

# Features
X = df.drop(['status', 'salary'], axis=1)

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# LOGISTIC REGRESSION MODEL
# ============================================

log_model = LogisticRegression()

log_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_model.predict(X_test_scaled)

# ---------- EVALUATION ----------

print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# ============================================
# RANDOM FOREST MODEL
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# ---------- EVALUATION ----------

print("\n===== RANDOM FOREST RESULTS =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)

# ---------- FEATURE IMPORTANCE GRAPH ----------

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")

plt.show()

# ============================================
# SAMPLE PREDICTION
# ============================================

sample_student = X.iloc[[0]]

prediction = rf_model.predict(sample_student)

if prediction[0] == 1:
    print("\nPrediction: Student is likely to be PLACED")
else:
    print("\nPrediction: Student is likely to NOT be placed")

# ============================================
# SAVE FEATURE IMPORTANCE
# ============================================

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature importance file saved!")

# ============================================
# END OF PROJECT CODE
# ============================================