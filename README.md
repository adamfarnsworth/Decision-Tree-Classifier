# Decision-Tree-Classifier
Creates a decision tree classifier for the
GoodMovie problem using the scikit-learn function DecisionTreeClassifier.
The file key.txt shows how features are represented as numeric values. (I.e.,
Budget=0 means Low, Budget=1 means Medium, etc.).  

The syntax of your program should be:  

```python
% def run_train_test (training_file, testing_file)
```  

The training_file and testing_file are file objects returnd by open() in
Python. Basically, you need to do 3 things: examine the txt files and convert them into
the desired input format, run DecisionTreeClassifier, calculate the results.
Use all of the class defaults, including the Gini function for the impurity measure, except to FIX
the random_state to 0 (otherwise your output may not be consistent in each run, you can
play around with it if you like but make sure you submit the results with it FIXED!)
Now modify your GoodMovie classifier to use entropy instead of Gini to measure
impurity. (This merely involves setting the criterion parameter of
DecisionTreeClassifier.) Run it on the same training and testing sets
(Be sure to notice if the change in impurity measure modified the results or not!)
As output, the program should print out the following numbers: the number of true
positives, true negatives, false positives, and false negatives, and the error rate, for the
two classifiers respectively, as shown here: (keys should match the exact strings!)  
```python
{
“gini”: {
"True positives” = 100, “True negatives” = 100,
“False positives” = 100, “False negatives”= 100,
“Error rate” = 0.99
}
“entropy”: {
"True positives” = 100, “True negatives” = 100,
“False positives” = 100, “False negatives”= 100,
“Error rate” = 0.99
}
}
```
