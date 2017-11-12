                                   Weird News Classification & Ranking

# Weird News Classification & Ranking
## What is a weird news?

These are the kind of news wich after reading them induces a sense of disbelief or alienation.

## Problem Statement : 

Figuring out the weirdness score and ranking weird or odd news stories.

![Weird News Classification & Ranking Image](https://ptpb.pw/wrdf.png)

## What we have done:
- Classification of the news i.e. whether it is weird or not.
- Comparison with existing Machine Learning techniques to provide a survey of the performance of various models for this task.
- Providing a novel Deep Learning architecture for this classification task.
- Classifiers used:  
  - Naive Bayes
  - Support Vectors Machines
  - Random forest classifiers
  - Gradient boosting classifiers
  - Ada boosting classifiers
  - Convolutional neural networks
  - Decision tree classifier
  - 3-layered-Perceptron
  - LSTM
  - Auto ML


## RANKING:
- The Ranking Task: Ranking of the weird news as per its weirdness score.
- Providing a way to rank a set of news based on their weirdness scores.
- Provide a metric that can be used to quantify the weirdness of a news.

## Dataset for ranking evaluation:
We have developed good quality annotations for a dataset that can be used as a gold standard for training and evalution of the ranking predictions. For annotations purposes, we have built a simple annotation application that can be used to gather annotations from the multiple annotators. This simplifies the process of gathering good quality annotations.  


## Probability for Ranking Task:
- The Weirdness Probability: Weirdness Probability is calculated with respect to the dataset which we have developed with annotation.
- We are using the models like Neural Net , Gaussian ‘s  Naive Bayes , Support Vector Machine, Random Forest , Ada Boost and Clustering for finding the probability.
- We are applying the Weirdness ranking based on this probability .
- We also explore a novel Affinity Propogation based clustering method for generating synthetic ranks for the dataset using clustering.

## Ranking Task:
The Ranking is briefly classified into 4 parts:
- Weirdness Probability > 75% will have the Weirdness score as 3
- Weirdness Probability > 50% & <= 75% will have the Weirdness score as 2
- Weirdness Probability > 25% & <=50%  will have the Weirdness score as 1
- Weirdness Probability > =0% & <=25% will have the Weirdness score as 0 


## Requirements:
python3, autoML, sklearn,gensim, keras, numpy

## How to run:
```python init.py <clasifier name>```  
e.g. python init.py nn

Classifiers name:  
```
nb = Naive Bayes  
svm = Support Vectors Machines  

rfc = Random Forest Classifier  
gbc = Graident Boosting Classifier  
abc = Adaboosting Classifier  
dt = Decision Tree Classifier  

nn = Neural Networks  
lstm = LSTM deep learning model  
cnn = CNN deep learning model  

automl = autoML classifier  
```


