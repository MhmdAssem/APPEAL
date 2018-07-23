import pandas as pd 
import numpy as np
import math
import collections

#df -> dataFrame

def ReadFiles():
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.drop(['id'],1,inplace=True)
	df = df.replace('?',np.NaN)
	df = df.apply(pd.to_numeric)
	df.fillna(df.mean(),inplace=True)
	#print(df.isnull().values.any())
	return df

def DivideTrainAndTest(df):
	train = 0.7
	train_size = (int)(df.shape[0]*0.7)
	df_train = df[0:train_size].copy()
	df_test = df[train_size:].copy()
	df_train_Class = np.array(df_train['Class'])
	df_test_Class = np.array(df_test['Class'])
	df_test.drop('Class',1,inplace=True)
	df_train.drop('Class',1,inplace=True)
	df_train = np.array(df_train)
	df_test = np.array(df_test)
	return df_train,df_train_Class,df_test,df_test_Class


	
df = ReadFiles()
df_train ,df_train_Class ,df_test ,df_test_Class = DivideTrainAndTest(df)




