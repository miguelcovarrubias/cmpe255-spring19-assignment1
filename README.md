# CMPE-255 Assignment 1


### Quick summary:
#### After trying different classifiers such as KNN I ended up using the SGD classifier as it gave me the best score results. For the data labeling I used known positive, negative and neutral words with different weights numbers (-5 to 5). [Words with labels](https://github.com/fnielsen/afinn/blob/master/afinn/data/AFINN-en-165.txt). Furthermore, it takes too long for the application to run when using all the data which means that the following examples will use a subset of the data.

### How to run the application:
```
By default the dataset to be used will be 1,000 yelp review records.  
To run the application from jupyter notebook, load assignment1.ipynb.
To run the application from python commandline, run python3 ass1_classifiers.py.
```

### Sample run with 10,000 yelp reviews. 

### PART 1

#### Data labeling results:
```
Positive entries: 7964
Negative entries: 794
Neutral entries: 1242
```
#### SGD Classifier with max iterations of results and using all test data. Results:

#### Scores:
```
SGD got accuracy score of 0.8395
SGD got cross validation score of [0.82871064 0.832021   0.82589118]
SGD got precision score of 0.6258920088401925
SGD got recall score of 0.5931373429100902
SGD got F1 score of 0.6078640411062998
```

##### Note: see function "def sgd_clf_fun(X_train, X_test, y_train, y_test):" for implemetation details.

### PART 2

#### SGD Classifier with max iterations of 100  nd using only the test data associated with 5 stars. Results:

#### Scores:

```
SGD got accuracy score of 0.9313953488372093
SGD got cross validation score of [0.82871064 0.832021   0.82589118]
SGD got precision score of 0.3333333333333333
SGD got recall score of 0.31046511627906975
SGD got F1 score of 0.3214930764599639
```

#### Comparizon ranking against Yelp's 5-stars ranking and share the instances that are opposite output (Yelp gives 5-stars and you classifier gives negative) as well as same output:

##### Instances where it predicted positive (and it should be positive): 
```
--> i just tried the chicken shawarma and it was awesome! very fresh and the flavors were on point. great customer service as well! i hope they come back to my neck of the woods soon!!
--> rami and borna ,very friendly and knowledgeable staff,i will recommend to everybody.thank you guys,great costumers service :).n.todorov
--> i would like to thank dr. banfield and the entire staff at apex endodontics for such an amazing visit! from the moment i walked in until the moment i left, i felt so comfortable and not anxious at all. thanks for a wonderful visit!
--> just get my chevy cruze 2012 fixed at classic gold. both michael and the secretary are super nice ! michael carefully examine the car and provided me a very reasonable estimate. they fixed the car as soon as they get the parts. everything runs very smoothly and i am very happy with their high standard service. i would definitely recommend it to my friend !
--> busy busy as usual. this location is in the heart of the strip. there's a line always but it goes quick!   the food is very fresh and the staff is extremely happy and kind it's actually very noticeable. staff is also always cleaning the tables and floors nonstop. very impressive!
```
##### Instances where it predicted negative (and it should be positive): 
```
--> they don't service north las vegas but ryan was very nice and called me back right away! would refer anyone to him!
--> for people living on the opposite side of the globe, before the internet was so widespread, television movies news were the source of information and the additions in the bucket list came from there too :)  ocean's 11 came in 2001, and in the end when everyone is just looking at the the bellgaio fountain .. it was then when it became an addition to the bucket list .. and i bet not just mine. then internet became widespread .. very common .. and then everyone was off to vegas. me!! the first time i came to america, the very first time .. it was due to work; in vegas :) .. i mean what better luck.  and without doubt .. the fountains of bellagio are one of those things which look even better in real life .. though off late things have become more flashy in reel life :p but not this one :)  it is just a great feeling to be able to stand there, right on the rails and watch those fountains dance ..  no, you won't come to vegas just to watch these fountains dance but if you are here i doubt if you would want to go past 'em without stopping :) and some people would rather skip "viva las vegas" to take a pic with the bootys for instagram or whatever .. i would not :)
--> i've shopped off and on at penney's my whole life. this store is the best. excellent customer service from the manager all the way to the cashiers. i recently encountered a frustrating situation with an online order, i called the national customer service number and it was a 13 minute wait. i called this ahwatukee store and the assistant manager took over and had the issue fixed immediately with extremely satisfying results. they have my loyalty.
--> don't treat this place as just your average donut shoppe! it's all that and a bag of chips (literally they sell chips too) not only do they offer scrumptious freshly made donuts and pastries & hot breakfast items but a wide variety of deli style sandwiches as well!!   i've been here so many times but i've yet to order lunch!? i tried a turkey & cheese on a buttery croissant- it was so big i couldn't finish it! i ordered the combo which came with a bag of chips, fountain drink and choice of donut. my sandwich was outstanding and i will be back to try more.
```
##### Instances where it predicted neutral (and it should be positive): 
```
--> our server dylan was phenomenal! he made my daughter feel incredibly special and made our lunch date amazing!!!
--> free ping pong and pool.  little more pricy than other bars but not bad for scottsdale.
--> très bons plats, portions généreuses, prix abordables. excellente valeur pour le prix. il peuvent améliorer le service.
--> by far the best mexican food we have had in las vegas. the fideo soup was loaded with flavor and so we're the salsas. we ordered the lunch combinations so we could try a variety and we were not disappointed by any of them. the service was impeccable, we will definitely be back soon.
--> best driving range on the west side of cleveland.  new balls every year, huge grass area that is kept up, putting green.  covered area with heaters for the bad weather as well.
```

### Sample KNN run using "def knn_clf_fun(X_train, X_test, y_train, y_test)"
```
KNN got accuracy score of 0.7925
```
