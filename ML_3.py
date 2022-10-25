#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.naive_bayes import GaussianNB


# Q1_
# 1.  (Titanic Dataset)
# 1. Find the correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case in class. 
# a. Do you think we should keep this feature? 
# 2. Do at least two visualizations to describe or show correlations. 
# 3. Implement Naïve Bayes method using scikit-learn library and report the accuracy

# In[7]:


train_df = pd.read_csv('C:/Users/deepp/OneDrive/Desktop/ML/Dataset/Dataset/train.csv')
test_df = pd.read_csv('C:/Users/deepp/OneDrive/Desktop/ML/Dataset/Dataset/test.csv')
combine = [train_df, test_df] #add data


# In[8]:


print(train_df.columns.values) #features of our database


# In[9]:


train_df.head() #check dataset 
#Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.


# In[10]:


train_df.tail() #check ending of dataset


# In[67]:


train_df.info()
print('_'*60)
test_df.info() # how to check two table info in one , only make one line between two table and run.


# In[13]:


train_df.describe()  # describe data


# In[68]:


##Analyze by pivoting features ---> the higher the number means more correlation with the target


# In[16]:


print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*20)
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*20)
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*20)
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# In[ ]:


#Analyze by visualizing data
#Correlating numerical features


# In[17]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[18]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# In[19]:


grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep',order=[1, 2, 3], hue_order=None)
grid.add_legend()


# In[20]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[21]:


train_df = train_df.drop(['Ticket', 'Cabin','Parch','SibSp'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin','Parch','SibSp'], axis=1)


# In[22]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[23]:


print(dataset['Title'])


# In[24]:


for dataset in combine:
     dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Jonkheer', 'Dona'], 'Lady')
     dataset['Title'] = dataset['Title'].replace(['Capt', 'Don', 'Major', 'Sir'], 'Sir')
     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
###Creating new feature extracting from existing


# In[25]:


title_mapping = {"Col": 1, "Dr": 2, "Lady": 3, "Master": 4, "Miss": 5, "Mr": 6, "Mrs": 7, "Rev": 8, "Sir": 9}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
###We can convert the categorical titles to ordinal


# In[26]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
##Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId


# In[27]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
##Converting a categorical feature


# In[28]:


print(train_df.isnull().sum()) #check any null value or not 


# In[29]:


train_df['Embarked'].describe()


# In[30]:


common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


# In[31]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[32]:


meanAge = int(train_df.Age.dropna().mean())
print('Mean Age = ', meanAge)


# In[33]:


for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(meanAge)
    dataset['Fare'] = dataset['Fare'].fillna(test_df['Fare'].dropna().median())


# In[34]:


combine[0].to_csv('train_preprocessed.csv',index=False)
combine[1].to_csv('test_preprocessed.csv',index=False)


# In[36]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[38]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:





# QUESTION 2:
# (Glass Dataset) 
# 1. Implement Naïve Bayes method using scikit-learn library.
# a. Use the glass dataset available in Link also provided in your assignment.
# b. Use train_test_split to create training and testing part. 
# 2. Evaluate the model on testing part using score and  classification_report(y_true, y_pred) 

# In[39]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings # current version generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")


# In[44]:


glass = pd.read_csv("C:/Users/deepp/OneDrive/Desktop/ML/Dataset/Dataset/glass.csv")


# In[45]:


glass.head()


# In[46]:


glass.describe()


# In[47]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[48]:


X = glass.iloc[:, :-1].values
y = glass.iloc[:, -1].values


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[56]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[57]:


# predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score check here
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:





# question:3 
# 1. Implement linear SVM method using scikit library 
# a. Use the glass dataset available in Link also provided in your assignment.
# b. Use train_test_split to create training and testing part. 
# 2. Evaluate the model on testing part using score and  classification_report(y_true, y_pred) 

# In[58]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[59]:


X = glass.iloc[:, :-1].values
y = glass.iloc[:, -1].values


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6, random_state = 0)


# In[66]:


from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:




