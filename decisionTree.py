from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def createTreeAndPredict(trainSet, testSet):

    # poner aqui los parametros y explicarlos
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', min_impurity_decrease=0.00003)

    # 1 to note into account id
    data = trainSet.iloc[:,1:12]
    test = trainSet.iloc[:,12]

    # discart parameters very related to each others
    data = data.drop(['F3'], axis=1)
    
    clf = clf.fit(data, test)

    testData = testSet.iloc[:,1:12]

    testData = testData.drop(['F3'], axis=1)

    return clf.predict(testData)



