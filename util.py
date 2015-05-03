import numpy as np
from numpy.core.umath import sign


def reshapeFeatureVector(train_X, columnWidths):
    newFeatureMatrix = np.zeros((train_X.shape[0], sum(columnWidths)))
    for rowIndex, featureVector in enumerate(train_X):
        newColumnIndex = 0
        for origColIndex, colWidth in enumerate(columnWidths):
            if colWidth == 1:
                newFeatureMatrix[rowIndex][newColumnIndex] = featureVector[origColIndex]
            else:
                for ii in range(int(colWidth)):
                    if (ii < len(featureVector[origColIndex])):
                        newFeatureMatrix[rowIndex][newColumnIndex + ii] = featureVector[origColIndex][ii]

            newColumnIndex += colWidth

    return newFeatureMatrix

def getFeatureColumnWidths(train_X):
    columnWidths = np.zeros(train_X.shape[1])

    for row in train_X:
        for index, col in enumerate(row):
            length = getLengthOfFeature(col)
            if columnWidths[index] < length:
                columnWidths[index] = length;
    return columnWidths

def getLengthOfFeature(feature):
    try:
        return len(feature)
    except TypeError:
        return 1

def ceilingForY(X, Y):
    changes = 0
    
    y = Y.ravel()
    lengths = [x[1] for x in X]
    y_signs = sign(y)
    y_vals = abs(y)
    
    for i in range(len(y)):
        if y_vals[i] > lengths[i]:
            y[i] = lengths[i]
            y[i] *= y_signs[i]
            changes += 1
            
    print str(changes) + " ceilings adjusted."
    return y
    