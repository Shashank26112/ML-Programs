import pandas as pd

data=pd.read_csv("C:/Users/MCA/Desktop/ML Lab Dataset/movies.csv")
print(type(data))

#print(data)
#print(data.head(3))
#print(data.tail())
#print(data.sample(3))
#print(data.sample(random_state=3))

# print(data.info())
# print(data.isnull().sum())

# Working with missing values
# data_1=data.dropna(axis=0,how="all")
# print(len(data))
# print(len(data_1))

# remove the data which any of the attribute  value is null in any one row  then delete entire row(only for refer)
# data_1=data.dropna(axis=0,how="any")
# print(len(data_1))

# delete the data any where for the particular attribute having less null vales
# data_1=data.dropna(axis=0,how="all",subset=["GENRE"])
# print(len(data_1))

#deleting the coloumn
# 1. deleting the Gross colum having the null values
data=data.drop(["Gross"],axis=1)
#print(data.isnull().sum())

#replaceing the values with null values

#replace the data 'votes' attribute having null values and fill with 0
#print(data['VOTES'])
data['VOTES']=data['VOTES'].fillna("0")
#print(data['VOTES'])
#print(data.isnull().sum())

# replace the  Runtime attribute having null values with the calculating the mean and put the avg values in place of null
#print(data['RunTime'])
meanRT=data['RunTime'].mean()
#print(meanRT)
meanRT=round(meanRT,1)
#print(meanRT)
data['RunTime']=data['RunTime'].fillna(meanRT)
#print(data['RunTime'])

#print(data.isnull().sum())

#replace the RATING attribute having null values with the calculating the mean and put the avg values in place of null
#print(data['RATING'])
meanRating=data['RATING'].mean()
#print(meanRating)
meanRating=round(meanRating,1)
#print(meanRating)
data['RATING']=data['RATING'].fillna(meanRating)
#print(data['RATING'])
#print(data.isnull().sum())

#for the GENRE

#print(data['GENRE'])
data['GENRE']=data['GENRE'].fillna("Comedy")
#print(data['GENRE'])
#print(data.isnull().sum())

#for the YEAR ATTRIBUTE
#print(data['YEAR'])
data['YEAR']=data['YEAR'].fillna(1990)
#print(data['YEAR'])

print(data.isnull().sum())
