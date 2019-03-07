# CMPE-255 Assignment 1


### Quick summary:
#### After trying different classifiers such as KNN I ended up using the SGD classifier as it gave me the best score results. For the data labeling I used known positive, negative and neutral words with different weights numbers (-5 to 5). [Words with labels](https://github.com/fnielsen/afinn/blob/master/afinn/data/AFINN-en-165.txt) 


You will be building a Yelp review classifier for our new ranking scheme:

Neutral (reviews with keywords like fine, OK etc..)
Positive (reviews with keywords like great, awesome etc..)
Negative (reviews with keywords like bad, awful etc..)
You can use any type of classifications (binary or multi) to classify the review dataset given by Yelp Open Dataset

Requirements
You must use Scikit Learn library only for this assignment. You need to submit these scores for your classifier.

Cross validation score
Precision
Recall
F1 score
Finally, you need to compare your ranking against Yelp's 5-stars ranking and share the instances that are opposite output (Yelp gives 5-stars and you classifier gives negative) as well as same output.