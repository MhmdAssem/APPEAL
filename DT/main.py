import pandas as pd 
import numpy as np
import math
import collections

#df -> dataFrame
# comments to be Removed
lovly=0
def ReadFiles(): # Read Data and remove any missing data 
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.drop(['id'],1,inplace=True)
	df = df.replace('?',np.NaN)
	df = df.apply(pd.to_numeric)
	df.fillna(df.mean(),inplace=True)
	#print(df.isnull().values.any())
	return df

def DivideTrainAndTest(df): # Divide Data to testing and Training
	train = 0.7
	train_size = (int)(df.shape[0]*0.7)
	df_train = df[0:train_size].copy()
	df_test = df[train_size:].copy()
	#df_train_Class = np.array(df_train['Class'])
	#df_test_Class = np.array(df_test['Class'])
	#df_test.drop('Class',1,inplace=True)
	#df_train.drop('Class',1,inplace=True)
	#df_train = np.array(df_train)
	#df_test = np.array(df_test)
	return df_train,df_test

def GetEntropy(df): # Calculate Entropy of Class Labels 
	Unique = pd.unique(df['Class'])
	Entropy=0.0
	for i in Unique:
		NumberOfRepeat=df.groupby('Class').get_group(i).count()[0]
		if NumberOfRepeat == 0:
			continue
		Entropy = Entropy + (-1.0 * NumberOfRepeat/df.shape[0]*np.log2(NumberOfRepeat/df.shape[0]))
	return Entropy
	
def SplitAndGetEntrpoy(df,UniqueValuesPerFeature,feature):#Calculate Entropy for the whole feature to choose which feature to split upon
	EntropyPerFeature=0.0
	for value in UniqueValuesPerFeature:
		df_feature = df.groupby(feature).get_group(value)
		EntropyPerFeature = EntropyPerFeature + (df.groupby(feature).get_group(value).count()[0]/df.shape[0])*GetEntropy(df_feature)
	return EntropyPerFeature
	
def GenerateDT(df,ParentEntropy = 1):# Decision Tree Algo
	
	MaxGain = -1.0
	FeatureSplit = ''
	tmp={}
	df = df.reset_index(drop = True)
	#Do a Fast Check to check if its a leaf node !
	if df.groupby('Class').count().shape[0] ==1:
		tmp['Class']=df['Class'][0]
		return tmp
	for feature in df.columns:
		if (feature == 'Class' ):
			continue
		UniqueValuesPerFeature = sorted(pd.unique(df[feature]))
		EntropyPerFeature =SplitAndGetEntrpoy(df,UniqueValuesPerFeature,feature)		
		FeatureGain = ParentEntropy- EntropyPerFeature
		if np.greater(FeatureGain , MaxGain) == True:
			MaxGain = FeatureGain
			FeatureSplit = feature
			
	if(FeatureSplit == ''):
		return None
	
	for value in sorted(pd.unique(df[FeatureSplit])):
		check = GenerateDT(df.groupby(FeatureSplit).get_group(value).drop(FeatureSplit,1),MaxGain)
		if check == None:
			continue
		else:
			tmp[value]=check
	
	return {FeatureSplit:tmp}

def CalculateAccurcy(df_test,Rules):
	Right=0
	Total=0
	Accurcy=0.0
	for index,row in df_test.iterrows():
		predictedLabel = MatchRules(row,Rules)
		if predictedLabel == row['Class']:
			Right+=1
		Total+=1
	Accurcy = Right/Total
	return Accurcy

def MatchRules(row,Rules,StartingFeature=''):
	
	for key in Rules:
		if key =='Class':
			return Rules[key]
		if StartingFeature == '':			
			return MatchRules(row,Rules[key],key)
		
		if key == row[StartingFeature]:
			return MatchRules(row,Rules[key],'')
	return None	
	
	
if __name__ == '__main__':	
	df = ReadFiles()
	df_train,df_test  = DivideTrainAndTest(df)
	Rules ={}
	Rules = GenerateDT(df)#Change it to df_train 100% sure to have an answer 
	Accurcy = CalculateAccurcy(df_test,Rules)
