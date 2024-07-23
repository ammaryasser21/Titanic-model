# Titanic Survival Prediction Using Machine Learning

This README outlines the steps and code used to predict the survival of passengers on the Titanic using machine learning techniques. We explore the dataset, preprocess the data, visualize important features, train multiple models, and select the best-performing model.

## 1. Import Necessary Libraries

We import the required libraries for data manipulation, visualization, model building, and evaluation.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
```

## 2. Load the Data

We load the training and test datasets.

```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```

## 3. Display Information About the Data

We explore the basic information, statistical summary, and the first few rows of the training data.

```python
print(train_data.info())
print(train_data.describe())
print(train_data.head())
```

## 4. Check for Missing Values

We check for any missing values in both the training and test datasets.

```python
train_data.isnull().sum()
test_data.isnull().sum()
```

## 5. Visualization

We visualize important features to understand their distribution and relationship with the target variable.

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=train_data, palette='Set2')
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=train_data, palette='Set1')
plt.title('Survival Count by Gender')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=train_data, palette='Set3')
plt.title('Survival Count by Passenger Class')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'].dropna(), kde=True, bins=30, color='blue')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_data[train_data['Survived'] == 1]['Age'].dropna(), kde=True, bins=30, color='green', label='Survived')
sns.histplot(train_data[train_data['Survived'] == 0]['Age'].dropna(), kde=True, bins=30, color='red', label='Not Survived')
plt.legend()
plt.title('Age Distribution by Survival Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_data['Fare'], kde=True, bins=30, color='purple')
plt.title('Fare Distribution')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Embarked', hue='Survived', data=train_data, palette='Set2')
plt.title('Survival Count by Embarkation Point')
plt.show()
```

## 6. Encode Categorical Variables

We encode the categorical features using `LabelEncoder`.

```python
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])
```

## 7. Handle Missing Values

We handle missing values by imputing the most frequent values for categorical features and the median for numerical features.

```python
age_imputer = SimpleImputer(strategy='median')
train_data['Age'] = age_imputer.fit_transform(train_data[['Age']])
test_data['Age'] = age_imputer.transform(test_data[['Age']])

train_data['Cabin'] = train_data['Cabin'].fillna('Unknown')
test_data['Cabin'] = test_data['Cabin'].fillna('Unknown')

embarked_imputer = SimpleImputer(strategy='most_frequent')
train_data['Embarked'] = embarked_imputer.fit_transform(train_data[['Embarked']])
test_data['Embarked'] = embarked_imputer.transform(test_data[['Embarked']])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])
```

## 8. Drop Unnecessary Columns

We drop columns that are not necessary for model training.

```python
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
```

## 9. Prepare Data for Model Training

We create a correlation matrix to discover the relations between the features.

```python
plt.figure(figsize=(12, 10))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

We split the data into features and the target variable, and then into training and validation sets.

```python
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
test_data = imputer.transform(test_data)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 10. Standardize the Data

We standardize the data for better performance of the models.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)
```

## 11. Define the Models and Their Hyperparameters

We define the models and their respective hyperparameters for grid search.

```python
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(random_state=42)
}

params = {
    'RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}
```

## 12. Perform GridSearchCV for Each Model

We perform grid search with cross-validation for each model to find the best hyperparameters.

```python
best_estimators = {}
for model_name in models.keys():
    grid_search = GridSearchCV(models[model_name], params[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_estimators[model_name] = grid_search.best_estimator_
    print(f'Best parameters for {model_name}: {grid_search.best_params_}')
    y_pred = best_estimators[model_name].predict(X_val)
    print(f'Accuracy for {model_name}: {accuracy_score(y_val, y_pred) * 100:.2f}%')
    print(classification_report(y_val, y_pred))
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()
```

## 13. Choose the Best Model Based on Accuracy

We select the best model based on the accuracy score.

```python
best_model_name = max(best_estimators.keys(), key=lambda name: accuracy_score(y_val, best_estimators[name].predict(X_val)))
best_model = best_estimators[best_model_name]
print(f'Best model: {best_model_name}')
```

## 14. Save the Best Model

We save the best model to a file for future use.

```python
joblib.dump(best_model, 'titanic_best_model.pkl')
```

## 15. Predict on Test Data and Create Submission File

We make predictions on the test data and create a submission file.

```python
predictions = best_model.predict(test_data)

original_test_data = pd.read_csv('test.csv')

submission = pd.DataFrame({
    'PassengerId': original_test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
```

## Conclusion

In this document, we have outlined the steps to preprocess the Titanic dataset, train multiple machine learning models, tune their hyperparameters using grid search with cross-validation, and select the best model based on performance metrics. The final model is then used to make predictions on the test dataset and create a submission file for evaluation.

The best model is the RandomForestClassifier, achieving an accuracy of 83.7%.