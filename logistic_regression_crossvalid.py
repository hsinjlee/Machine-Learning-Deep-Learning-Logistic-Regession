import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn import cross_validation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


creditData = pd.read_csv("/Users/robert/Documents/Pycharm/logistic_regression/credit_data.csv")
# creditData = pd.read_csv("C:\\Users\\User\\Desktop\\credit_data.csv")

features = creditData[["income","age","loan"]]
target = creditData.default

model = LogisticRegression()
# predicted = cross_validation.cross_val_predict(model,features,target, cv=10)
predicted = cross_val_predict(model,features,target, cv=10)

print()
print("accuracy_score(target,predicted)")
print(accuracy_score(target,predicted))

#
accuracy_score(target,predicted)
0.9195
#