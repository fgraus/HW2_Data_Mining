import pandas as pd
from handleData import getTrainningData
from decisionTree import createTreeAndPredict
from naiveBayes import createModelAndPredict

def main():
    return 0


def test():
    trainData = getTrainningData()
    test = getTrainningData('Resources/test.csv')

    predictions = createModelAndPredict(trainData, test)

    file = open('Resources/pruebas.dat','w')

    for i in range(predictions.shape[0]):
        if(predictions[i] == 1):
            file.write('1\n')
        else:
            file.write('0\n')

    file.close()

def train():

    trainData = getTrainningData()

    NUMBEROFCROSSVALIDATIONS = 4

    scoreSum = 0

    for i in range(0,NUMBEROFCROSSVALIDATIONS):
        posStart = i * round(trainData.shape[0]/NUMBEROFCROSSVALIDATIONS)
        possEnd = (i + 1) * round(trainData.shape[0]/NUMBEROFCROSSVALIDATIONS)

        test = trainData.iloc[posStart:possEnd]
        train = trainData.iloc[:posStart]
        train = train.append(trainData.iloc[possEnd:])

        data = pd.DataFrame({'Real': test.iloc[:,12], 'Prediction': createModelAndPredict(train, test)})

        scoreSum += computeScore(data)


    print('Total f-score: ' + str(scoreSum/NUMBEROFCROSSVALIDATIONS))
    return 0

def computeScore(data):

    allPositivesDetected = data.loc[data.Real == data.Prediction].loc[data.Real == 1].shape[0]
    allPositivesPredictes = data.loc[data.Prediction == 1].shape[0]
    precision = allPositivesDetected/allPositivesPredictes

    falseNegatives = data.loc[data.Real == 1].loc[data.Prediction ==0].shape[0]
    recall = allPositivesDetected / (allPositivesDetected + falseNegatives)

    f1_score = 2 * (precision*recall)/(precision + recall)

    return f1_score


test()