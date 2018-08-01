import pandas as pd 
import numpy as np
import math
import collections

Global_Constant = 0 # try to solve problem from the start :D

def ReadFiles():
	df = pd.read_csv('nursery.data.txt')
	return df

def DivideTrainAndTest(df):
	train = 0.7
	train_size = (int)(df.shape[0]*train)
	df_train = df[0:train_size].copy()
	df_test = df[train_size:].copy()

	return df_train,df_test
	
def Calculate_Number_Of_Occuerance_For_Each_Label(df_train,Column,Value,Class_values_Repeatness):#|Class_Labels
	global Global_Constant
	Number_Of_Occuerance = {}
	unique_Values_Of_Class = pd.unique(df_train[df_train.columns[-1]])
	for Class_Value in unique_Values_Of_Class:
		if Class_Value in pd.unique(df_train.groupby(Column).get_group(Value)[df_train.columns[-1]]):
			Number_Of_Occuerance[Class_Value] =(df_train.groupby(Column).get_group(Value).groupby(df_train.columns[-1]).get_group(Class_Value).count()[0])/Class_values_Repeatness[Class_Value] + Global_Constant
		else:
			Number_Of_Occuerance[Class_Value] =Global_Constant
			
	return Number_Of_Occuerance
	
def Calculate_Probs_Of_Unique_values_in_Column(df_train,Column,Class_values_Repeatness):#Get_Every_Value_In_Column_To_get_TheProbabilites
	Probs_Of_Unique_values_in_Column ={}
	unique_Values_Of_Column = pd.unique(df_train[Column])
	for Value in  unique_Values_Of_Column:
		Probs_Of_Unique_values_in_Column[Value]=Calculate_Number_Of_Occuerance_For_Each_Label(df_train,Column,Value,Class_values_Repeatness)
		
	return Probs_Of_Unique_values_in_Column
	
def Calculate_Probabilites(df_train):#Generate Table of probabilites for each Column|Class_Label
	Probs = {}
	unique_Values_Of_Class = pd.unique(df_train[df_train.columns[-1]])
	Class_values_Repeatness={}
	for label in unique_Values_Of_Class:
		Class_values_Repeatness[label] = df_train.groupby(df_train.columns[-1]).get_group(label).count()[0]
		
	for Column in df_train.columns:
		if Column == 'Class':
			continue
		Probs[Column]=Calculate_Probs_Of_Unique_values_in_Column(df_train,Column,Class_values_Repeatness)
	
	Columns_Unique_Values ={}
	for col in df_train.columns:
		if col != 'Class':
			Columns_Unique_Values[col] = pd.unique(df_test[col]).shape[0]
	
	return Probs,Class_values_Repeatness,Columns_Unique_Values,df_train.shape[0]
		
def Iterate_Over_All_Columns(Probs,Class_values_Repeatness,Columns_Unique_Values,Column_Label= ' ',Dimension = 0):#help to solve zero problem by Recurse over all dict
	for k,v in Probs.items():
		if isinstance(v,dict):
			if Dimension == 0:
				Column_Label = k
			Probs[k] = Iterate_Over_All_Columns(v,Class_values_Repeatness,Columns_Unique_Values,Column_Label,Dimension+1)
		else:
			if v == 0:
				Probs =Find_This(Probs,Class_values_Repeatness[k],Columns_Unique_Values[Column_Label],k)
	
	return Probs

def Find_This(Probs,Class_Repeat,Column_Distinct_Value,Class_Label):#find particular columns given label
	for k,v in Probs.items():
		if isinstance(v,dict):
			Probs[k] = Find_This(v,Class_Repeat,Column_Distinct_Value,Column_Label,Class_Label)
		else:
			if k == Class_Label:
				Probs[k] = Solve_Zero_Problem(v,1/Column_Distinct_Value,Class_Repeat)
	
	return Probs
				
def Solve_Zero_Problem(nc,p,n):
        return (nc + p)/ (n+1)

def Search_For_Value(Probs,Column,Label):
	for k,v in Probs.items():
		if k == Column:
			return Search_For_Value(v,Column,Label)
		if k == Label:
			return v
	
def Does_It_Match(Probabilities,Test_Class,Class_values_Repeatness,Total): #multiply by each class and get label baby 
	MaxProb = -1
	Choosen_Label = ''
	for k,v in Probabilities.items():
		tmp = (v)* (Class_values_Repeatness[k]/Total)
		
		if np.greater(tmp , MaxProb) == True:
			MaxProb = tmp
			Choosen_Label =k
		
	if Choosen_Label == Test_Class:
		return 1
	else:
		return 0
	
def Test(df_test,Probs,Class_values_Repeatness,Total):
	Total_Trials = 0
	Right = 0
	for index,row in df_test.iterrows():
		Probabilities ={}
		for Column in df_test.columns:
			if Column == 'Class':
				continue
			Classes=Search_For_Value(Probs,Column,row[Column])
			if Classes == None:
				continue
			for k,v in Classes.items():
				if k in Probabilities:
					Probabilities[k] = Probabilities[k]*v
				else:
					Probabilities[k] = v
		Right+= Does_It_Match(Probabilities,row['Class'],Class_values_Repeatness,Total)
		Total_Trials+=1
	return Right/Total_Trials


def Print_Dict(Dict):
	for k,v in Dict.items():
		if isinstance(v,dict):
			Print_Dict(v)
			print(' ')
		else:
			print("{0} : {1} ".format(k,v))
	
	
if __name__ == '__main__':	
	df = ReadFiles()
	df = df.sample(frac=1,random_state=np.random.RandomState())
	df_train,df_test  = DivideTrainAndTest(df)
	Probs ,Class_values_Repeatness,Columns_Unique_Values,Total=Calculate_Probabilites(df_train)
	Probs = Iterate_Over_All_Columns(Probs,Class_values_Repeatness,Columns_Unique_Values)
	#print(Probs)
	Accuracy = Test(df_test,Probs,Class_values_Repeatness,Total)
