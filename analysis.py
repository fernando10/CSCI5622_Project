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
    
def outputUserAnalysis(D, categoryNames, filename):
    categoryIndices = range(len(categoryNames))
    
    f = open(filename, "w")
    f.write("user")
    for c in categoryIndices:
        f.write(",abs(" + categoryNames[c] + ")")
    for c in categoryIndices:
        f.write(","+categoryNames[c])
    for c in categoryIndices:
        f.write(",abs(Predicted " + categoryNames[c] + ")")
    for c in categoryIndices:
        f.write(",Predicted "+categoryNames[c])
    f.write("\n")
    
    for ukey in sorted(D.keys()):
        f.write(str(ukey))
        for i in [0,1]:
            for c in categoryIndices:
                f.write(",")
                if c in D[ukey][i].keys():
                    avg = sum([abs(x) for x in D[ukey][i][c]]) / len(D[ukey][i][c])
                    f.write(str(avg))
            for c in categoryIndices:
                f.write(",")
                if c in D[ukey][i].keys():
                    avg = sum(D[ukey][i][c]) / len(D[ukey][i][c])
                    f.write(str(avg))

        f.write("\n")

def avgBuzzPerUserByCategory(X, Y, predicted_Y, categoryNames, filename):
    D = dict()
    users = [int(x[0]) for x in X]
    categories = [int(x[2]) for x in X]
    y = Y.ravel()
    
    for i in range(len(users)):
        if users[i] not in D:
            D[users[i]] = (dict(), dict())
        if categories[i] not in D[users[i]][0]:
            D[users[i]][0][categories[i]] = []
            D[users[i]][1][categories[i]] = []
            
        D[users[i]][0][categories[i]].append(y[i])
        D[users[i]][1][categories[i]].append(predicted_Y[i])

    outputUserAnalysis(D, categoryNames, filename)

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
    
