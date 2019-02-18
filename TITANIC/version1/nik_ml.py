# Titanic Datasets
# Python imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
# reading the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()
train_df.info()
train_df.describe()
# let check the null values
# 38% survived in titanic.
total = train_df.isnull().sum().sort_values(ascending=False)
print total.head()
len(train_df)
percent1.head()
missing_data = pd.concat([total,percent1], axis=1,keys=['Total','%'], sort=False)
missing_data.head()
train_df.columns.values
# drop the passengerID
train_df.drop(['PassengerId'], axis=1,inplace=True)
test_df.drop(['PassengerId'], axis=1,inplace=True)
train_df.columns.values
train_df[["Cabin","Survived"]].head()
import re
deck = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'U':8}
data = [train_df,test_df]
for value  in data:
    value["Cabin"] = value["Cabin"].fillna('U0')
    value["Deck"] = value["Cabin"].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    value["Deck"] = value["Deck"].map(deck)
    value["Deck"] = value["Deck"].fillna(0)
    value["Deck"] = value["Deck"].astype('int')
    #print value
    #value["Cabin"]
value.head()
# check the null values
total = train_df.isnull().sum().sort_values(ascending=False)

# drop the Cabin values
train_df.drop(['Cabin'], axis=1,inplace=True)
test_df.drop(['Cabin'], axis=1,inplace=True)
train_df.columns.values
data = [train_df, test_df]
for value in data:
    mean = value["Age"].mean()
    std = value["Age"].std()
    is_null = value["Age"].isnull().sum()
    print 'Mean:-- {}, STD:-- {}, NULL:-- {}'.format(mean,std,is_null)
    # compute the random values
    rand_age = np.random.randint(mean-std,mean + std,size=is_null)
    # print rand_age
    # fill the null values
    age_slice = value["Age"].copy()
    #print age_slice
    age_slice[np.isnan(age_slice)] = rand_age
    #print age_slice
    value["Age"] = age_slice
    value["Age"] = value["Age"].astype(int)
train_df.head()
# Now the number of null values are:--
train_df.isnull().sum().sort_values(ascending=False)

train_df["Embarked"].head()
train_df["Embarked"].describe()
# replace the null value with the common value 'S'
data = [train_df, test_df]
for value in data:
    value["Embarked"] = value["Embarked"].fillna('S')
train_df.isnull().sum().sort_values(ascending=False)

train_df.info()
train_df["Fare"].head(10)
data = [train_df, test_df]
for value in data:
    value['Fare'] = value['Fare'].fillna(value["Fare"].median())
    value['Fare'] = value['Fare'].astype(int)
train_df["Sex"].head()

gender = {'male':0, 'female':1}
data = [train_df, test_df]
for value in data:
    value["Sex"] = value["Sex"].map(gender)

train_df.head(10)
             # dropping the Name features
train_df.drop(['Name'],inplace=True, axis=1)
test_df.drop(['Name'],inplace=True, axis=1)
train_df.info()
train_df["Embarked"].head()
values = {'S':0,'C':1,'Q':2}
data = [train_df,test_df]
for value in data:
    value["Embarked"] = value["Embarked"].map(values)
train_df["Ticket"].describe()
train_df.drop(['Ticket'],axis=1,inplace=True)
test_df.drop(['Ticket'],axis=1,inplace=True)
# Training the data into models
Y_train = train_df["Survived"]
X_train = train_df.drop(["Survived"], axis=1)
X_test.head()
test_df.columns.values
# Python imports from sklearn
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# Random Forest
regressor = RandomForestClassifier(n_estimators=20, random_state = 0)
regressor.fit(X_train, Y_train)
X_test = X_test.drop(["Name"],axis=1)
X_test.head()
regressor.predict(X_test)
