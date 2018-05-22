import numpy
#from sklearn import datasets
from sklearn import tree
from sklearn.metrics import confusion_matrix

def run_train_test(training_file, testing_file):

    # Budget Genre FamousActors Director GoodMovie
    next(training_file)
    next(testing_file)
    trainingData = []
    testingData = []

    # getting all raw data
    for line in training_file:
        trainingData.append(map(int,line.split()))
    for value in testing_file:
        testingData.append(map(int,value.split()))

    
    trainingSamples = trainingData  # samples
    trainingLabels = []             # labels
    # sepearating sample and label values from raw data
    for sample in trainingSamples:
        sample.pop(0)
        trainingLabels.append(sample.pop(-1))   

    # sklearning magic GO!
    # gini
    gini = tree.DecisionTreeClassifier(random_state=0)
    gini = gini.fit(trainingSamples, trainingLabels)
    # entropy
    entropy = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
    entropy = entropy.fit(trainingSamples, trainingLabels)

    testingSamples = testingData
    testingLabels = []
    for sample in testingSamples:
        sample.pop(0)
        testingLabels.append(sample.pop(-1))

    #TN, FP, FN, TP
    gtn, gfp, gfn, gtp = confusion_matrix(testingLabels, gini.predict(testingSamples)).ravel()
    etn, efp, efn, etp = confusion_matrix(testingLabels, entropy.predict(testingSamples)).ravel()
    #error rate = FP+FN/ P+N
    PplusN = len(testingLabels)
    ger = float((float(gfp+gfn)/float(PplusN)))
    eer = float((float(efp+efn)/float(PplusN)))

    # creating out Dictionary
    giniDic = {'True positives': gtp, 'True negatives': gtn,'False positives': gfp,'False negatives': gfn,'Error rate': ger}
    entropyDic = {'True positives': etp, 'True negatives': etn,'False positives': efp,'False negatives': efn,'Error rate': eer}
    finalDic = {'gini':giniDic, 'entropy':entropyDic}

    #print finalDic
    return finalDic
    pass


if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()