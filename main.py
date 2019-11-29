import numpy as np
import collections

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from JSONUpdater import JSONUpdater
from ImportDataset import ImportDataset
from SVM import SVM
from Preprocessor import Preprocessor

import csv 
import re 

label_positive = 1 
label_negative = 0

create_json = JSONUpdater()
svm = SVM()

np.random.seed(500)

Encoder = LabelEncoder()

preproc = Preprocessor('export_dataframe.csv')
preproc.process()

corpusFinal = preproc.getCorpusTextFinal()
corpusLabel = preproc.getCorpusLabel()

print(corpusFinal)

# train and test dataset split 
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpusFinal,corpusLabel,test_size=0.2)

# label encoding
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
realData_Y = Encoder.fit_transform(preproc.getCorpusLabel())

# get percentage of positive and negative sample in training sample 
train_count_pos_neg = collections.Counter(Train_Y)
train_pos_percentage = (train_count_pos_neg[label_positive]/len(Train_Y))*100
train_neg_percentage = (train_count_pos_neg[label_negative]/len(Train_Y))*100

# get percentage of positive and negative sample in test sample 
test_count_pos_neg = collections.Counter(Test_Y)
test_pos_percentage = (test_count_pos_neg[label_positive]/len(Test_Y))*100
test_neg_percentage = (test_count_pos_neg[label_negative]/len(Test_Y))*100

# get percentage of positive and negative sample from real data
real_count_pos_neg = collections.Counter(realData_Y)
real_pos_percentage = (real_count_pos_neg[label_positive]/len(realData_Y))*100
real_neg_percentage = (real_count_pos_neg[label_negative]/len(realData_Y))*100

Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(corpusFinal)

print()
print("Size of m : ")
print("----------------------------")
print(len(Tfidf_vect.get_feature_names()))
print()

print("jumlah data training : ")
print(len(Train_X))
print()

print("jumlah data test : ") 
print(len(Test_X))
print() 

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
AllData_X_Tfidf = Tfidf_vect.transform(preproc.getCorpusText())

print("DTM Data Latih")
print(Train_X_Tfidf)

print("DTM Data Uji")
print(Test_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
svm.fit(Train_X_Tfidf,Train_Y)

print("get support vector : ",svm.get_support_vector())


# predict the labels on validation dataset
predictions_SVM = svm.predict(Test_X_Tfidf)

# predict all data 
predictions_all = svm.predict(AllData_X_Tfidf)

# prediction test set
print('prediction test : ', predictions_SVM)

# Use accuracy_score function to get the accuracy
print("prediction test accuracy -> ",accuracy_score(predictions_SVM, Test_Y)*100)

# prediction test set
print('prediction all : ', predictions_all)

# Use accuracy_score function to get the accuracy
print("prediction all accuracy -> ",accuracy_score(predictions_all, realData_Y)*100)

# get percentage of positive and negative sample in test after svm
test_after_svm_count_pos_neg = collections.Counter(predictions_SVM)
test_after_svm_pos_percentage = (test_after_svm_count_pos_neg[label_positive]/len(predictions_SVM))*100
test_after_svm_neg_percentage = (test_after_svm_count_pos_neg[label_negative]/len(predictions_SVM))*100  

# get percentage of positive and negative sample all after svm
all_svm_count_pos_neg = collections.Counter(predictions_all)
all_svm_pos_percentage = (all_svm_count_pos_neg[label_positive]/len(predictions_all))*100
all_svm_neg_percentage = (all_svm_count_pos_neg[label_negative]/len(predictions_all))*100  

outputJson = svm.mergeJSON(predictions_all, preproc.getCorpusJson())
#print(outputJson)

conf_matrix = confusion_matrix(Test_Y, predictions_SVM)
print("Confusion Matrix : ") 
print(conf_matrix)

create_json.write_result(outputJson)
create_json.set_percentage(
        real_pos_percentage,
        real_neg_percentage,
        train_pos_percentage,
        train_neg_percentage,
        test_pos_percentage,
        test_neg_percentage,
        test_after_svm_pos_percentage,
        test_after_svm_neg_percentage,
        all_svm_pos_percentage,
        all_svm_neg_percentage)

