import pandas as pd 
import numpy as np
import math
import collections

#df -> dataFrame

def ReadFiles():
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.drop(['id'],1,inplace=True)
	df.replace('?',-9999999,inplace=True)#Too Big Value :D
	return df

def DivideTrainAndTest(df):
	train = 0.7
	train_size = (int)(df.shape[0]*0.7)
	df_train = df[0:train_size]
	df_test = df[train_size:]
	df_train_Class = np.array(df_train['Class'])
	df_test_Class = np.array(df_test['Class'])
	df_test.drop('Class',1,inplace=True)
	df_train.drop('Class',1,inplace=True)
	df_train = np.array(df_train)
	df_test = np.array(df_test)
	return df_train,df_train_Class,df_test,df_test_Class

def GetPrediction(Map,k):
	Max_Repeat =0
	predicted_Label=''
	Min_Distance =1000000.0
	for key in Map:
		if Map[key] <= k and Map[key]>Max_Repeat:
			Max_Repeat      = Map[key] 
			Min_Distance    = key[0]
			k               = k - Max_Repeat
			predicted_Label = key[1]
			
	return predicted_Label	
	
	
def KNN(df_train,df_train_Class,df_test,df_test_Class):

	for k in range(1,9):
		Total_Trials =0
		Right_Trials =0
		Predicted = np.array([-1])
		for row_test in range(0,df_test.shape[0]):
			Map={}#(distance,class):occuerance
			Eq_distance =0.0
			for row_train in range(0,df_train.shape[0]):
				for col in range (0,df_test.shape[1]):
					Eq_distance = Eq_distance + ((float)(df_test[row_test][col])-(float)(df_train[row_train][col]))**2
				
				
				Eq_distance = math.sqrt(Eq_distance)
				check_If_exist_before =(Eq_distance,df_train_Class[row_train]) in Map
				if check_If_exist_before == False:
					Map[(Eq_distance,df_train_Class[row_train])] = 1
				else:
					Map[(Eq_distance,df_train_Class[row_train])] = Map[(Eq_distance,df_train_Class[row_train])]+1
			Map=collections.OrderedDict(sorted(Map.items()))		
			Prediction = GetPrediction(Map,k)
			#Prediction = np.array(Prediction)
			#Predicted =np.append(Predicted,Prediction)
			if Prediction == df_test_Class[row_test]:
				Right_Trials = Right_Trials+1
			Total_Trials= Total_Trials+1
		print(k,Right_Trials/Total_Trials)
		#print(Predicted)
					
					
					
df = ReadFiles()
df_train ,df_train_Class ,df_test ,df_test_Class = DivideTrainAndTest(df)
KNN(df_train ,df_train_Class ,df_test ,df_test_Class)




