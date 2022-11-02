import pandas as pd
from sklearn.utils import shuffle

#id - unique identifier
#F1 - Continuous value describing number of years since last degree was completed
#F2 - Continuous value indicating hours worked per week
#F3 - Categorical Value
#F4 - Categorical Value indicating type of occupation
#F5 - continuous value denoting gains
#F6 - continuous value denoting loss
#F7 - Categorical value denoting marital status
#F8 - Categorical value denoting type of employment (e.g., Self)
#F9 - Categorical Value denoting education type
#F10 - Categorical Value denoting different race
#F11 - Categorical - Female/Male
#credit - 0: Bad, 1: Good


def getTrainningData(path = 'Resources/train.csv'):

    trainData = pd.read_csv(path)
    
    trainData.F10 = pd.Categorical(trainData.F10)
    trainData.F11 = pd.Categorical(trainData.F11)
    
    trainData.F10 = trainData.F10.cat.codes
    trainData.F11 = trainData.F11.cat.codes

    trainData.F1 = pd.cut(trainData.F1,8, labels=[1,2,3,4,5,6,7,8])
    trainData.F2 = pd.cut(trainData.F2,8, labels=[1,2,3,4,5,6,7,8])
    trainData.F5 = pd.cut(trainData.F5,8, labels=[1,2,3,4,5,6,7,8])
    trainData.F6 = pd.cut(trainData.F6,8, labels=[1,2,3,4,5,6,7,8])

    #trainData = trainData.sample(frac=1)

    return trainData

getTrainningData()