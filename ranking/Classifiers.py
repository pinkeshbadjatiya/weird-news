from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
#import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
#from init import get_dataset
import pdb
import math

target_names=['NormalNews','WeirdNews']

def ranking(prob):
	if prob<0.25 :
	    return 0
	elif prob<0.5 :
	    return 1
	elif prob<0.75 :
	    return 2
	else:
	    return 3

def rankingList(prob_list, ytrue):
	weirdprob=[]
	for i in xrange(len(prob_list)):
	    #weirdprob.append(ranking(prob_list[i][1]))
	    weirdprob.append(prob_list[i][1])
        #pdb.set_trace()
        print "="*20
        print "- Ranking Results -"
        print "MSE:", mean_squared_error(ytrue, weirdprob)
        print "RMSE:", math.sqrt(mean_squared_error(ytrue, weirdprob))


def classify(clf,Traindata,TrainLabels,test,labels_test):
        #pdb.set_trace()
	clf.fit(Traindata,TrainLabels)
	#p=clf.predict(test)
	#acc = accuracy_score(labels_test, p)
	#print "accuracy : ",acc
	#print "Precision & Recall"
	#print classification_report(labels_test, p, target_names=target_names)

def Gaussian_NB(Traindata,TrainLabels,test,labels_test):
	clf=GaussianNB()
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob = clf.predict_proba(test)
	rankingList(prob, labels_test)

def lr(Traindata,TrainLabels,test,labels_test):
	clf=LogisticRegression()
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob= clf.predict_proba(test)
	rankingList(prob, labels_test)

def Decision_tree(Traindata,TrainLabels,test,labels_test):
	clf=tree.DecisionTreeClassifier()
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob= clf.predict_proba(test)
	rankingList(prob, labels_test)

def SVM_predict(Traindata,TrainLabels,test,labels_test,val=1.0,Kern='rbf'):
	clf = SVC(kernel=Kern, C=val, probability=True)
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob= clf.predict_proba(test)
	rankingList(prob, labels_test)

def randomforest_predict(Traindata,TrainLabels,test,labels_test):
	clf = RandomForestClassifier(max_depth=10,random_state=0)
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob= clf.predict_proba(test)
	rankingList(prob, labels_test)

def XGBoost(Traindata,TrainLabels,test,labels_test):
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob= clf.predict_proba(test)
	rankingList(prob, labels_test)


def ADABoost(Traindata,TrainLabels,test,labels_test):
	clf = AdaBoostClassifier(n_estimators=100)
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob= clf.predict_proba(test)
	rankingList(prob, labels_test)

def NN(Traindata,TrainLabels,test,labels_test):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, 50, 50), random_state=1)
	classify(clf,Traindata,TrainLabels,test,labels_test)
	prob= clf.predict_proba(test)
	rankingList(prob, labels_test)

def autoML(Traindata,TrainLabels,test,labels_test):
        clf = autosklearn.classification.AutoSklearnClassifier()
	classify(clf,Traindata,TrainLabels,test,labels_test)




