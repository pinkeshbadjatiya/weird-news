                                          IRE Course Team Project - Monsoon 2017

# Weird News Classification & Ranking
## What is a weird news?

These are the kind of news wich after reading them induces a sense of disbelief or alienation.

## Problem Statement : 
Figuring out the weirdness score and ranking weird or odd news stories.

## What we have done:
![Weird News Classification & Ranking Image](https://ptpb.pw/wrdf.png)
- **Classsification Task**
  - Classification of the news i.e. whether it is weird or not.
  - Comparison with existing Machine Learning techniques to provide a survey of the performance of various models for this task.
  - Providing a novel Deep Learning architecture for this classification task.
- **Data Annotation**
  - Developed a good quality dataset for the ranking task that can be used as a gold standard for evaluation.
- **Ranking Task**
  - Ranking of the weird news as per its weirdness score.
  - Providing a way to rank a set of news based on their weirdness scores.
  - Provide a metric that can be used to quantify the weirdness of a news.

## CLASSIFICATION
### Classifiers used: 
  - Machine Learning based methods
    - Naive Bayes
    - Support Vectors Machines
    - Random forest classifiers
    - Gradient boosting classifiers
    - Ada boosting classifiers
    - Decision tree classifier
  - Deep Learning based methods
    - Convolutional neural networks
    - 3-layered-Perceptron
    - LSTM
    - Auto ML
  - Heuristic based methods
    - Affinity propogation based clustering
    - KNN clustering

## DATA ANNOTATION:  
- We have developed good quality annotations for a dataset that can be used as a gold standard for training and evalution of the ranking predictions. For annotations purposes, we have built a simple annotation application that can be used to gather annotations from the multiple annotators. This simplifies the process of gathering good quality annotations.  

## RANKING:
### The Ranking Task
- Ranking of the weird news as per its weirdness score.
- Providing a way to rank a set of news based on their weirdness scores.
- Provide a metric that can be used to quantify the weirdness of a news.

### Probability for Ranking Task:
- The Weirdness Probability: Weirdness Probability is calculated with respect to the dataset which we have developed with annotation.
- We are using the models like Neural Net , Gaussian ‘s  Naive Bayes , Support Vector Machine, Random Forest , Ada Boost and Clustering for finding the probability.
- We are applying the Weirdness ranking based on this probability .
- We also explore a novel Affinity Propogation based clustering method for generating synthetic ranks for the dataset using clustering.

### Procedure to Rank:  
Since we do not have very large annotation set for ranking (just 500 samples), we perform a attempt to learn
the level of weirdness using the binary-classified samples.
1. We learn a classifier using the binary classification task. This allows the model to predict the probability
of a news article being weird or not.
2. We then use that model and to predict the bucket in which the news article belongs.
3. We then test our model based on the gold standard labels for the test samples. We use the RMSE
metric to compare the performance of various methods.
4. Gold Standard Labels: For our task we take gold standard label as the average of the annotations
provided by the annotators, scaled in 0 to 1 range.


## Ranking Task:
The Ranking is briefly classified into 4 parts:
- Weirdness Probability > 75% will have the Weirdness score as 3
- Weirdness Probability > 50% & <= 75% will have the Weirdness score as 2
- Weirdness Probability > 25% & <=50%  will have the Weirdness score as 1
- Weirdness Probability > =0% & <=25% will have the Weirdness score as 0 


### Clustering Based Ranking model:  
We also experiment with a clustering based ranking method. Here is the procedure that
we follow for our method:
1. We use the LSTM based model to learn a classifier for the task.
2. We use the output of the intermediate layer to project a new test sample in the vector space of 300
dimensions.
3. We then cluster all the train samples in that vector space using Affinity Propagation based clustering
algorithm and label a cluster as {0, 1, 2, 3} based on the majority of the points in the cluster.
4. Now, for any given point in test dataset, we find its cluster and assign that label to the point.

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


