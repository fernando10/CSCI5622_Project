import numpy as np
import pdb

def reshapeFeatureVector(featureVector, columnWidths):
    newFeatureMatrix = np.zeros((featureVector.shape[0], sum(columnWidths)))
    for rowIndex, featureVector in enumerate(featureVector):
        newColumnIndex = 0
        for origColIndex, colWidth in enumerate(columnWidths):
            if colWidth == 1:
                if not isinstance(featureVector[origColIndex],int) and not isinstance(featureVector[origColIndex],float) and not isinstance(featureVector[origColIndex],long):
                    pdb.set_trace()
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

