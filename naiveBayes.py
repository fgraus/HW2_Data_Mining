from sklearn.naive_bayes import CategoricalNB

def createModelAndPredict(trainSet, testSet):
    clf = CategoricalNB(alpha=0.000003)

    # 1 to note into account id
    data = trainSet.iloc[:,1:12]
    test = trainSet.iloc[:,12]

    # discart parameters very related to each others
    data = data.drop(['F3'], axis=1)
    
    clf = clf.fit(data, test)

    testData = testSet.iloc[:,1:12]

    testData = testData.drop(['F3'], axis=1)

    return clf.predict(testData)

def categoricalData(data):
    return data


