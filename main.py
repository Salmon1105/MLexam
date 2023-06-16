import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.ticker as ticker
from sklearn import preprocessing

#notice: Disable all warnings
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('loan_test.csv')
df.head()
df.shape
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()
df['loan_status'].value_counts()
# notice: installing seaborn might takes a few minutes


bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df[['Principal','terms','age','Gender','education']].head()
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

X = Feature
X[0:5]
y = df['loan_status'].values
y[0:5]

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test)
accuracy = []
for k in range(1, 10):
    Knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree")

    Knn.fit(X_train, y_train)
    y_pred = Knn.predict(X_test)
    accuracy.append((100 * accuracy_score(y_test, y_pred)))
plt.plot(accuracy)
Knn = KNeighborsClassifier(n_neighbors=8,algorithm="kd_tree")
Knn.fit(X_train, y_train)
print("Accuracy of 8NN for test",100*accuracy_score(y_test,y_pred),"%")
print("Accuracy of 8NN for train",100*accuracy_score(y_train,Knn.predict(X_train)),"%")
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(X_train, y_train)
y_pred = DecisionTree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for train = ",accuracy_score(y_train, DecisionTree.predict(X_train))*100,"%")
print("Accuracy for test = ",accuracy*100,"%")
from sklearn import svm
SVM = svm.SVC(kernel='linear')
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for train = ",accuracy_score(y_train, SVM.predict(X_train))*100,"%")
print("Accuracy for test = ",accuracy_score(y_test,y_pred)*100,"%")
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for train = ",accuracy_score(y_train, LR.predict(X_train))*100,"%")
print("Accuracy for test = ", accuracy_score(y_test, y_pred)*100,"%")
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

test_df = pd.read_csv('loan_test.csv')
test_df.head()
def preProcessing(df):
  df['due_date'] = pd.to_datetime(df['due_date'])
  df['effective_date'] = pd.to_datetime(df['effective_date'])
  df['dayofweek'] = df['effective_date'].dt.dayofweek
  df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
  df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
  Feature = df[['Principal','terms','age','Gender','weekend']]
  Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
  Feature.drop(['Master or Above'], axis = 1,inplace=True)
  Feature = preprocessing.StandardScaler().fit(Feature).transform(Feature)
  X = Feature
  y = df['loan_status'].values
  return X, y
X, y = preProcessing(test_df)
def evaluate(model, X, y):
  print(f"Jaccard: {jaccard_score(y, model.predict(X), pos_label='PAIDOFF')}")
  print(f"F1: {f1_score(y, model.predict(X), pos_label='PAIDOFF')}")
evaluate(Knn, X, y)
evaluate(DecisionTree, X, y)
evaluate(SVM, X, y)
evaluate(LR, X, y)
print(f"log loss: {log_loss(y, LR.predict_proba(X))}")