
dataset:
three .txt files i.e. 
-traindata {labelled data}
-testdata
-testlabels

how to run:

copy the appropiate dataset in the .txt files
& 
python init.py <clasifier name>
e.g. python init.py nn

classifiers name :
nb=naivebayes
svm=support vectors machines

rfc=randomforestclassifiers
gbc=graidentboostingclassifiers
abc=adaboostingclassifiers
dt=decisiontreeclassifier

nn = neural networks
lstm=LSTM deep learning model
cnn=CNN deep learning model

automl = autoML classifier
