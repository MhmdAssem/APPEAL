import pandas as pd 
import numpy as np
import math
import collections
PI = 3.14159265359
# Normlization normalized_df=(df-df.mean())/df.std()
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
	train_size = (int)(df.shape[0]*train)
	df_train = df[0:train_size].copy()
	df_test = df[train_size:].copy()
	return df_train,df_test

def Normlize_Data(df):#Normlization not used
	for Column in df.columns:
		if Column == 'Class':
			continue
		df[Column] = (df[Column].mean())/df[Column].std()
	return df
	
def Calculate_Mean_and_Std(df):
	Mean ={}
	Std  ={}
	Uni_Values_Of_Classes = sorted(pd.unique(df[df.columns[-1]]))
	for Column in df.columns:
		if Column == 'Class':
			continue
		df_Column_Grp = df[Column].groupby(df['Class'])
		for Class_Label,Data  in df_Column_Grp:			
			Col_Mean ={}
			Col_Std  ={}
			Check_If_Column_Exist_Mean = Column in Mean
			Check_If_Column_Exist_std = Column in Std
			
			#Get Mean 
			if Check_If_Column_Exist_Mean == True:
				Col_Mean = Mean[Column]
			Col_Mean[Class_Label] = Data.mean()
			
			if Check_If_Column_Exist_std == True:
				Col_Std = Std[Column]
			Col_Std[Class_Label]  = Data.std()
			
			Mean[Column] = Col_Mean
			Std[Column] = Col_Std
		
		
	return Mean,Std,Uni_Values_Of_Classes

def Search_For_Missing_Labels(Mean,Std,Labels): # Check Missing Labels
	for label in Labels:
		#print (' ')
		for k,v in Mean.items():
			if label in v == False:
				dic = v
				dic[label] = 0
				Mean[k] = dic
		
		for k,v in Std.items():
			if label in v == False:
				dic = v
				dic[label] = 0
				Std[k] = dic
			
	return Mean,Std

	
def Calculate_The_Prob(Feature_Mean,Feature_Std,Xi):
	
	if Feature_Std == 0:
		return 1
	global PI
	return (1/(Feature_Std * math.sqrt(2*PI))) * np.exp(-1* ( (Xi - Feature_Mean)**2)/(2 * (Feature_Std**2) ) )

	
def Test(df_Test,Mean,Std,Labels):#Test the Algorithm
	
	Right = 0
	Total = 0
	for index,row in df_test.iterrows():
		predicted_label =''
		Max_Probability = -1
		for label in Labels:
			Label_Prob =1
			for Column in df_test.columns:
				if Column == 'Class':
					continue
				Feature_Mean = Mean[Column][label]
				Feature_Std  = Mean[Column][label]
				Label_Prob*=Calculate_The_Prob(Feature_Mean,Feature_Std,row[Column])
			
			if np.greater(Label_Prob , Max_Probability) == True:
				Max_Probability = Label_Prob
				predicted_label = label
		Total+=1
		if predicted_label == row['Class']:
			Right+=1
	return Right/Total

	
if __name__ == '__main__':	
	df = ReadFiles()
	df_train,df_test = DivideTrainAndTest(df)
	Mean,Std,Labels = Calculate_Mean_and_Std(df_train)
	Mean , Std = Search_For_Missing_Labels(Mean,Std,Labels)
	acc = Test(df_test,Mean,Std,Labels)
