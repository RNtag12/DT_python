# DT_python
Desision Tree
# Loan Data Analysis and Predictive Modeling

This project details the process of analyzing loan data and building predictive models to determine whether loans are likely to be fully paid. The analysis includes data preprocessing, visualization, and the application of machine learning models using Decision Tree and Random Forest classifiers.

## Table of Contents
1. Introduction
2. Data Overview
3. Data Analysis and Visualization
4. Data Preprocessing
5. Model Building and Evaluation
6. Conclusion

## 1. Introduction
Loan default prediction is a crucial task in the finance industry. Accurate predictions help lenders minimize risks and make informed decisions. This project aims to explore a dataset of loan applications, analyze the features, and build predictive models to classify loans as fully paid or not fully paid.

## 2. Data Overview
We begin by loading and examining the dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv('loan_data.csv')
data.head()
```

### Data Structure

```plaintext
   credit.policy          purpose  int.rate  installment  log.annual.inc    dti  fico  days.with.cr.line  revol.bal  revol.util  inq.last.6mths  delinq.2yrs  pub.rec  not.fully.paid
0             1  debt_consolidation   0.1189      829.10       11.350407  19.48   737         5639.958333     28854       52.1              0            0       0             0
1             1         credit_card   0.1071      228.22       11.082143  14.29   707         2760.000000     33623       76.7              0            0       0             0
2             1  debt_consolidation   0.1357      366.86       10.373491  11.63   682         4710.000000      3511       25.6              1            0       0             0
3             1  debt_consolidation   0.1008      162.34       11.350407   8.10   712         2699.958333     33667       73.2              1            0       0             0
4             1         credit_card   0.1426      102.92       11.299732  14.97   667         4066.000000      4740       39.5              0            1       0             0
```

### Data Summary

```python
data.info()
data.describe()
```

### Data Columns

```python
data.columns
```

## 3. Data Analysis and Visualization
We visualize the data to gain insights into the distribution and relationships between variables.

```python
fig = plt.figure(figsize=(10,6))
sns.pairplot(data=data[['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line']], hue='credit.policy', diag_kind='hist')
sns.set_style('whitegrid')
sns.countplot(data=data, x='credit.policy', palette='coolwarm')
sns.boxplot(x='credit.policy', y='installment', data=data)
sns.jointplot(x='int.rate', y='fico', data=data, kind='hex')

fig = plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True)

plt.figure(figsize=(10,6))
sns.countplot(data=data, x='purpose', hue='not.fully.paid')
```

## 4. Data Preprocessing
We preprocess the data by scaling numeric features and encoding categorical variables.

```python
from sklearn.preprocessing import StandardScaler

# Encode categorical variable 'purpose'
df = pd.get_dummies(data=data, columns=['purpose'], drop_first=True)

# Standardize numeric features
scaler = StandardScaler()
numeric_features = df.drop(['credit.policy', 'not.fully.paid'], axis=1).columns
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Separate features and target variable
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']
```

## 5. Model Building and Evaluation
We split the data into training and testing sets and apply two machine learning models: Decision Tree and Random Forest.

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

### Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

### Model Performance

- **Decision Tree Classifier**: Achieved 73% accuracy with moderate precision and recall for both classes.
- **Random Forest Classifier**: Achieved 84% accuracy with higher precision but lower recall for the minority class.

## 6. Conclusion
The Random Forest model outperforms the Decision Tree in terms of accuracy. However, it struggles with recall for the minority class (loans not fully paid). Further tuning and possibly additional data preprocessing or feature engineering may improve model performance.
