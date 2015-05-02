import math
import numpy as np
from numpy.core.umath import sign

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
