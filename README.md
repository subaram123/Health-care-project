# Health-care-project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Dataset
data = pd.read_csv('heart.csv')
print("📊 Dataset Loaded:")
print(data.head())

# Step 3: Check for Missing Values
print("\n🔍 Null Value Check:")
print(data.isnull().sum())

# Step 4: Encode Categorical Columns
le = LabelEncoder()
for col in ['cp', 'restecg', 'slope', 'thal']:
    data[col] = le.fit_transform(data[col])

# Step 5: Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("📌 Correlation Heatmap")
plt.show()

# Step 6: Split Features and Target
X = data.drop('target', axis=1)
y = data['target']

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Step 8: Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Step 9: Predictions and Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Decision Tree Accuracy: {accuracy:.2f}")

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot()
plt.title("📉 Confusion Matrix - Decision Tree")
plt.show()

# Step 11: ROC Curve
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('📈 ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()

# Step 12: Age vs Target (Top 3 High-Risk Age Groups)
risk_group = data.groupby('age')['target'].mean().reset_index()
top_risk = risk_group.sort_values(by='target', ascending=False).head(3)

print("\n🔥 Top 3 High-Risk Age Groups (based on % with disease):")
for i, row in top_risk.iterrows():
    print(f"Age {int(row['age'])} → Risk Chance: {round(row['target']*100, 2)}%")

# Step 13: Visualization – Heart Disease by Sex
plt.figure(figsize=(6,4))
sns.barplot(x='sex', y='target', data=data)
plt.xticks([0,1], ['Female', 'Male'])
plt.title("💔 Heart Disease Rate by Sex")
plt.ylabel("Proportion with Disease")
plt.show()
