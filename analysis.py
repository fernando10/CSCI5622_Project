import math
import numpy as np
from numpy.core.umath import sign
from pandas.core.common import intersection

def calcRMS(prediction, actual):
    if (len(actual) == 0):
        return 0
    x = 0.0
    for i in range(0,len(actual)):
            x += (prediction[i] - actual[i])**2
    x /= (len(actual))
    return math.sqrt(x)

def signMismatchPercentage(prediction,actual):
    if (len(actual) == 0):
        return 0
    x = 0.0
    for i in range(0,len(actual)):
        x += (sign(prediction[i]) == sign(actual[i]))    
    return x*100/len(actual)

def calcRMSPerCategory(prediction,actual,features,categories):
    print ("%15s%9s%18s"%("Category","RMS","+/- Accuracy"))
    for index, category in enumerate(categories):
        questionIndices = [i for i in range(0,len(actual)) if (features[i][2] == index)]
        predictionsOfCategory = prediction[questionIndices]
        actualsOfCategory = actual[questionIndices]
        rmsForCategory = calcRMS(predictionsOfCategory,actualsOfCategory)
        signAccuracy = signMismatchPercentage(predictionsOfCategory,actualsOfCategory)
        print ("%15s%10.2f%12.2f%%" % (category,rmsForCategory,signAccuracy))
        
def doAnalysis(prediction, actual,features,categories):
    print "------------Analysis---------------"
    absRms = calcRMS(abs(prediction),abs(actual))
    print ("Ignoring correctness of answer, obtained an RMS of: %.2f" % absRms)
    rms = calcRMS(prediction,actual)
    print ("Obtained a true RMS of: %.2f" % rms)
    signAccuracy = signMismatchPercentage(prediction,actual)
    print ("Predicted Correct Sign %.2f%% of the time." % signAccuracy)
    
    calcRMSPerCategory(prediction,actual,features,categories)

    print "------------End Analysis-----------"
    
def outputUserAnalysis(D, categories):
    categories.sort()
    
    f = open("user_analysis.csv", "w")
    f.write("user")
    for c in categories:
        f.write(",abs(" + str(c) + ")")
    for c in categories:
        f.write(","+str(c))
    f.write("\n")
    
    for ukey in sorted(D.keys()):
        f.write(str(ukey))
        for c in categories:
            f.write(",")
            if c in D[ukey].keys():
                avg = sum([abs(x) for x in D[ukey][c]]) / len(D[ukey][c])
                f.write(str(avg))
        for c in categories:
            f.write(",")
            if c in D[ukey].keys():
                avg = sum(D[ukey][c]) / len(D[ukey][c])
                f.write(str(avg))

        f.write("\n")

def avgBuzzPerUserByCategory(X, Y):
    D = dict()
    users = list(X['user'])
    categories = list(X['category'])
    y = list(Y['position'])
    
    for i in range(len(users)):
        if users[i] not in D:
            D[users[i]] = dict()
        if categories[i] not in D[users[i]]:
            D[users[i]][categories[i]] = []
            
        D[users[i]][categories[i]].append(y[i])    

    outputUserAnalysis(D, list(set(categories)))

def userQuestionIntersection(train, test):
    trainUsers = list(set(train['user']))
    testUsers = list(set(train['user']))
    userIsect = []
    for u in trainUsers:
        if u in testUsers:
            userIsect.append(u)
            
    trainQuestions = list(set(train['question']))
    testQuestions = list(set(train['question']))
    questionIntersect = []
    for q in trainQuestions:
        if q in testQuestions:
            questionIntersect.append(u)

    print "Number of unique users in train set: " + str(len(trainUsers))
    print "Number of unique users in test set: " + str(len(testUsers))
    print "Number of users in both sets (Intersection): " + str(len(userIsect))
    print "Number of unique questions in train set: " + str(len(trainQuestions))
    print "Number of unique questions in test set: " + str(len(testQuestions))
    print "Number of questions in both sets (Intersection): " + str(len(questionIntersect))
    
