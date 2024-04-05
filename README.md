# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Find the entropy of the tree when each field is taken as a root node
2. Consider the field with lowest entropy as root node and repeat the same with remaining nodes as subnode
4. End the tree when there are no more fields to divide and the classification is not mixed
5. Predict the output by traversing through the decision tree built 

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Bhargava S
RegisterNumber:  212221040029 
```python
import pandas as pd
data = pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/Bhargava-Shankar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/85554376/99d55078-4cac-4d60-ab1e-28954dc733de)
![image](https://github.com/Bhargava-Shankar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/85554376/584d3a3a-53ab-40be-964f-c3a13e2847ea)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
