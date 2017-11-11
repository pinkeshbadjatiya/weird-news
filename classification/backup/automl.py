import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.metrics import classification_report
import numpy as np
from init import get_dataset

target_names=['X','Y']
#iris = datasets.load_iris()
# print (iris)
#X, y = iris.data, iris.target
#print(X)
#data=len(X)
#print (X)
#print (data)
#print (y)
#Traindata=X[np.r_[0,1:data-50],:] 
#TrainLabels=y[np.r_[0,1:data-50]]
# print (Traindata)
# print("----------")
# print (TrainLabels)
#Testlabels=y[np.r_[0,50:data]]
#Testdata=X[np.r_[0,50:data],:] 

X, y, Xtest, Ytest = get_dataset()

clf = autosklearn.classification.AutoSklearnClassifier()
pdb.set_trace()
clf.fit(X, y)
#clf.fit(digits.data[:-1], digits.target[:-1])  
#print(clf.predict(Xtest)
acc = accuracy_score(clf.predict(Xtest), Ytest)
print (acc)




