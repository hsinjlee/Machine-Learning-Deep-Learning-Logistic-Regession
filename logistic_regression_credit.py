import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


creditData = pd.read_csv("C:\\Users\\User\\Desktop\\credit_data.csv")
print("creditData.head--------------------------------")
print()
print(creditData.head())
print()
print("creditData.describe--------------------------------")
print()
print(creditData.describe())
print()
print("creditData.corr--------------------------------")
print()
print(creditData.corr())

features = creditData[["income","age","loan"]]
target = creditData.default

feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)
predictions = model.fit.predict(feature_test)

print()
print("confusion matrix")
print(confusion_matrix(target_test, predictions))
print()
print("accuracy score")
print(accuracy_score(target_test,predictions))