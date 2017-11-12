       Weird News Classification & Ranking


What is a weird news?

- These are the kind of news wich after reading them induces a sense of disbelief or alienation.

Problem Statement : 

Figuring out the weirdness score and ranking weird or odd news stories.

What we have done:
Classification of the news i.e. whether it is weird or not.
Comparison with existing Machine Learning techniques to provide a survey of the performance of various models for this task.
Providing a novel Deep Learning architecture for this classification task.
Classifiers used:
Naive bayes
support vectors machines
Random forest classifiers
Gradient boosting classifiers
Ada boosting classifiers
Convolutional neural networks
Decision tree classifier
3-layered-Perceptron
LSTM
Auto ML

RANKING:
The Ranking Task:

Ranking of the weird news as per its weirdness score.
Providing a way to rank a set of news based on their weirdness scores.
Provide a metric that can be used to quantify the weirdness of a news.

Dataset for ranking evaluation:
We have developed a good quality dataset that can be used as a gold standard for evaluation.

Probability for Ranking Task:
The Weirdness Probability :
Weirdness Probability is calculated with respect to the dataset which we have developed with annotation.
We are using the models like Neural Net , Gaussian ‘s  Naive Bayes , Support Vector Machine, Random Forest , Ada Boost and Clustering for finding the probability.
We are applying  the Weirdness ranking based on this probability .

Ranking Task:
The Ranking is briefly classified into 4 parts:

Weirdness Probability > 75% will have the Weirdness score as 3
Weirdness Probability > 50% & <= 75% will have the Weirdness score as 2
Weirdness Probability > 25% & <=50%  will have the Weirdness score as 1
Weirdness Probability > =0% & <=25% will have the Weirdness score as 0 
