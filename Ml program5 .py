import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

#print(sns.get_dataset_names())
data=sns.load_dataset("titanic")
#print(data.head())
#print(len(data))
#print(data.info())

#print(data['survived'].value_counts())
#sns.countplot(x=data['survived'])
#plt.show()

# sns.countplot(x=data['survived'],hue=data['pclass'])
# plt.show()

# sns.countplot(x=data['survived'],hue=data['sex'])
# plt.show()

# sns.countplot(x=data['survived'],hue=data['alone'])
# plt.show()

#print(data.columns)
cols=['fare', 'class', 'who', 'adult_male', 'deck', 'embark_town',
       'alive', 'alone']
data_new=data.drop(cols,axis=1)
# print(data_new.head())
# print(data_new.isnull().sum())

meanAge=data_new['age'].mean()
meanAge=round(meanAge,2)
#print(meanAge)

data_new['age']=data_new['age'].fillna(meanAge)
#print(data_new.isnull().sum())

data_new=data_new.dropna()
# print(data_new.isnull().sum())
# print(len(data_new))

#print(data_new.info())

# print(data_new['sex'].value_counts())
# print(data_new['embarked'].value_counts())

#converting string to int using label encoding
enc=LabelEncoder()
data_new['sex']=enc.fit_transform(data_new['sex'])
data_new['embarked']=enc.fit_transform(data_new['embarked'])
#print(data_new.head())

#Features and Target
x=np.array(data_new.iloc[:,1:])
#print(x.shape)
y=np.array(data_new.iloc[:,0])
#print(y.shape)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8, random_state=3)
#print(pd.DataFrame(ytrain).value_counts())

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3,p=2)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
# print(ypred)
# print(ytest)

count=0
for i in range(len(ytest)):
       if ypred[i]==ytest[i]:
              count+=1

print(count)
print(count/len(ytest))
a=accuracy_score(ytest,ypred)
print(a)

#save the model
joblib.dump(model,"titanic.pkl")

mymodel=joblib.load("C:/Users/MCA/PycharmProjectsRaksML/program1scrapping/.venv/Scripts/titanic.pkl")
print(mymodel.predict([[1,0,20,2,0,2]]))
