import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import joblib

sns.set(style="whitegrid")
plt.rcParams['figure.figsize']=(10,6)

df=pd.read_csv("loan_status_data.csv")

df.head()
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

print(df.columns)
df["loan_status"].value_counts()
(df["loan_status"].value_counts(normalize=True)*100).round(2)


'''Age Distribution'''

plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Distribution of Applicant Age")
plt.xlabel("Age")
plt.show()

'''Income Distribution'''

plt.figure(figsize=(8,5))
sns.histplot(df["income"], bins=30, kde=True)
plt.title("Income Distribution")
plt.show()

'''Loan Amount Distribution'''
plt.figure(figsize=(8,5))
sns.histplot(df["loan_amount"], bins=30, color="orange", kde=True)
plt.title("Loan Amount Distribution")
plt.show()

'''Interest Rate Distribution'''

plt.figure(figsize=(8,5))
sns.histplot(df["rate"], bins=25, color="green", kde=True)
plt.title("Interest Rate Distribution")
plt.show()

'''Employment Length'''

plt.figure(figsize=(8,5))
sns.histplot(df["emp_length"], bins=20, color="purple", kde=True)
plt.title("Employment Length")
plt.show()

'''Credit History Length'''

plt.figure(figsize=(8,5))
sns.histplot(df["person_cred_hist_length"], bins=20, color="brown", kde=True)
plt.title("Credit History Length")
plt.show()

'''Home Ownership'''

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="home")
plt.title("Home Ownership")
plt.xticks(rotation=45)
plt.show()

'''Loan Intent'''

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="loan_intent")
plt.xticks(rotation=45)
plt.title("Purpose of Loan")
plt.show()

'''Loan Grade'''

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="loan_grade")
plt.title("Loan Grade")
plt.show()

'''Previous Default'''

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="person_default")
plt.title("Previous Default")
plt.show()

'''Loan Status'''

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="loan_status")
plt.title("Loan Status")
plt.show()

'''Outlier Detection for Income, Loan Amount, Interest Rate, Age'''

plt.figure(figsize=(8,4))
sns.boxplot(x=df["income"])
plt.title("Income Boxplot")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=df["loan_amount"])
plt.title("Loan Amount Boxplot")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=df["rate"])
plt.title("Interest Rate Boxplot")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=df["age"])
plt.title("Age Boxplot")
plt.show()

'''Bivariate Analysis'''

'''Income vs Loan Status'''

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="loan_status", y="income")
plt.title("Income vs Loan Status")
plt.show()

'''Age vs Loan Status'''

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="loan_status", y="age")
plt.title("Age vs Loan Status")
plt.show()

'''Loan Amount vs Loan Status'''

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="loan_status", y="loan_amount")
plt.title("Loan Amount vs Loan Status")
plt.show()

'''Interest Rate vs Loan Status'''

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="loan_status", y="rate")
plt.title("Interest Rate vs Loan Status")
plt.show()

'''Home Ownership vs Loan Status'''

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="home", hue="loan_status")
plt.xticks(rotation=45)
plt.title("Home Ownership vs Loan Status")
plt.show()

'''Loan Intent vs Loan Status'''

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="loan_intent", hue="loan_status")
plt.xticks(rotation=45)
plt.title("Loan Intent vs Loan Status")
plt.show()

'''Loan Grade vs Loan Status'''

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="loan_grade", hue="loan_status")
plt.title("Loan Grade vs Loan Status")
plt.show()

'''Previous Default vs Loan Status'''

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="person_default", hue="loan_status")
plt.title("Previous Default vs Loan Status")
plt.show()

'''Correlation Analysis'''

from sklearn.preprocessing import LabelEncoder
temp = df.copy()
le = LabelEncoder()
for col in temp.select_dtypes(include="object"):
    temp[col] = le.fit_transform(temp[col])

plt.figure(figsize=(12,8))
sns.heatmap(temp.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

'''Pairplot'''

sns.pairplot(
    df[["age","income","loan_amount","rate","loan_status"]],
    hue="loan_status"
)
plt.show()



'''Feature Engineering'''

df["income_loan_ratio"] = df["income"] / df["loan_amount"]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = [
    "home",
    "loan_intent",
    "loan_grade",
    "person_default"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

'''Train-Test Split'''

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

'''Feature scaling'''

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for col in df.columns:
    print(col, df[col].dtype)
for col in df.columns:
    print(col, df[col].unique()[:10])

# Replace all '?' in the dataframe with NaN
df.replace("?", np.nan, inplace=True)

# Convert numeric columns
df["emp_length"] = pd.to_numeric(df["emp_length"])
df["rate"] = pd.to_numeric(df["rate"])

# Fill missing numeric values with median
df["emp_length"].fillna(df["emp_length"].median(), inplace=True)
df["rate"].fillna(df["rate"].median(), inplace=True)

print(df.dtypes)
print(df.isnull().sum())

# Fill missing values with the median
df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())
df["rate"] = df["rate"].fillna(df["rate"].median())

print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = [
    "home",
    "loan_intent",
    "loan_grade",
    "person_default"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
print(df.dtypes)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train and Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Scaling the Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))
accuracy = pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy":[
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)

    ]
})
print(accuracy)

plt.figure(figsize=(7,5))
sns.barplot(data=accuracy,
            x="Model",
            y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.8,1)
plt.show()

best_model = rf
best_pred = rf_pred

print(classification_report(y_test, best_pred))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

y_prob = best_model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", round(auc,3))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr,
         label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],
         linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance")
plt.show()

import joblib
joblib.dump(rf, "loan_approval_model.pkl")

