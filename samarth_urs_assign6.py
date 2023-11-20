
#task 1
import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm

from operator import add
from pyspark import SparkContext

def calculate_freq_array(listOfIndices, numberofwords):
    returnVal = np.zeros(20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

def build_zero_one_array(listOfIndices):
    returnVal = np.zeros(20000)
    for index in listOfIndices:
        if returnVal[index] == 0:
            returnVal[index] = 1
    return returnVal

def compute_average_tf(word, data, dictionary):
    pos = dictionary.filter(lambda x: x[0] == word).take(1)[0][1]
    average_tf = data.map(lambda x: (x[1][pos])).mean()
    return average_tf

def display_average_tf_values(data, dictionary, label):
    print(f'Average TF values for {label} documents:')
    words = ['applicant', 'and', 'attack', 'protein', 'court']
    for word in words:
        average_tf = compute_average_tf(word, data, dictionary)
        print(f'Word: {word}, Average TF: {average_tf}')

sc = SparkContext(appName="LogRegression")

d_corpusTrain = sc.textFile(sys.argv[1])

numberOfDocsTrain = d_corpusTrain.count()

d_keyAndTextTrain = d_corpusTrain.map(lambda x: (x[x.index('id="') + 4 : x.index('" url=')],
                                                 x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWordsTrain = d_keyAndTextTrain.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))

allWords = d_keyAndListOfWordsTrain.flatMap(lambda x: x[1]).map(lambda x: (x, 1))
allCounts = allWords.reduceByKey(add)
topWords = allCounts.top(20000, lambda x: x[1])
topWordsK = sc.parallelize(range(20000))
dictionary = topWordsK.map(lambda x: (topWords[x][0], x)).cache()

allWordsWithDocIDTrain = d_keyAndListOfWordsTrain.flatMap(lambda x: ((j, x[0]) for j in x[1]))
allDictionaryWordsTrain = dictionary.join(allWordsWithDocIDTrain)
justDocAndPosTrain = allDictionaryWordsTrain.map(lambda x: (x[1][1], x[1][0]))
allDictionaryWordsInEachDocTrain = justDocAndPosTrain.groupByKey()
allDocsAsNumpyArraysTrain = allDictionaryWordsInEachDocTrain.map(lambda x: (x[0], calculate_freq_array(x[1], len(x[1]))))
allDocsAsNumpyArraysTrain.cache()

zeroOrOneTrain = allDocsAsNumpyArraysTrain.map(lambda x: (x[0], build_zero_one_array(np.where(x[1] > 0)[0])))
dfArrayTrain = zeroOrOneTrain.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
idfArrayTrain = np.log(np.divide(np.full(20000, numberOfDocsTrain), dfArrayTrain))
allDocsAsNumpyArraysTFidfTrain = allDocsAsNumpyArraysTrain.map(lambda x: (1 if 'AU' in x[0] else 0, np.multiply(x[1], idfArrayTrain)))
allDocsAsNumpyArraysTFidfTrain.cache()

wikiTF = allDocsAsNumpyArraysTrain.filter(lambda x: 'AU' not in x[0])
AusTF = allDocsAsNumpyArraysTrain.filter(lambda x: 'AU' in x[0])

display_average_tf_values(wikiTF, dictionary, 'Wikipedia')
display_average_tf_values(AusTF, dictionary, 'Australian Court')

#task 2

def logistic_regression_iteration(r, lr, precision, lam, xy_data, cost_old):
    i = 0

    while True:

        cost = xy_data.map(lambda x: -x[0] * np.dot(r, x[1]) + np.log(1 + np.exp(np.dot(r, x[1])))
                           ).reduce(add) + lam * np.sqrt(sum(r**2))


        deriv = (xy_data.map(lambda x: ((-x[1] * x[0]) + (x[1] * np.exp(np.dot(r, x[1])) /
                                                            (1 + np.exp(np.dot(r, x[1])))))
                              ).reduce(add)) + (2 * lam * r)


        if i >= 1:
            if cost < cost_old:
                lr = lr * 1.05

            else:
                lr = lr * 0.5


        if abs(cost - cost_old) <= precision:
            print('Cost stopped decreasing at:', i)
            break

        cost_old = cost


        r = r - lr * deriv

        i += 1

    return r

r = np.repeat(0, 20000)
lr = 0.001
precision = 0.1
cost_old = 0
lam = 0.005


def top_related_words(weights, dictionary):
    top_words = dictionary.filter(lambda x: x[1] in (np.argsort(weights)[-5:])).collect()
    print('Top 5 words most strongly related with an Australian court case:\n', top_words)

import time
start_time = time.time()
xy_data = allDocsAsNumpyArraysTFidfTrain

# Call the function
weights = logistic_regression_iteration(r, lr, precision, lam, xy_data, cost_old)

end_time = time.time()
print(f"Time taken for training: {end_time - start_time} seconds")
top_related_words(weights, dictionary)

#task 3
d_corpusTest = sc.textFile(sys.argv[2])
numberOfDocsTest = d_corpusTest.count()

d_keyAndTextTest = d_corpusTest.map(lambda x: (x[x.index('id="') + 4 : x.index('" url=')],
                                              x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWordsTest = d_keyAndTextTest.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))

allWordsWithDocIDTest = d_keyAndListOfWordsTest.flatMap(lambda x: ((j, x[0]) for j in x[1]))
allDictionaryWordsTest = dictionary.join(allWordsWithDocIDTest)
justDocAndPosTest = allDictionaryWordsTest.map(lambda x: (x[1][1], x[1][0]))

allDictionaryWordsInEachDocTest = justDocAndPosTest.groupByKey()
allDocsAsNumpyArraysTest = allDictionaryWordsInEachDocTest.map(lambda x: (x[0], calculate_freq_array(x[1], len(x[1]))))
allDocsAsNumpyArraysTest.cache()

zeroOrOneTest = allDocsAsNumpyArraysTest.map(lambda x: (x[0], build_zero_one_array(np.where(x[1] > 0)[0])))
dfArrayTest = zeroOrOneTest.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

idfArrayTest = np.log(np.divide(
    np.full(20000, numberOfDocsTest), dfArrayTest, out=np.zeros_like(dfArrayTest), where=dfArrayTest != 0),
    out=np.zeros_like(dfArrayTest), where=dfArrayTest != 0)

allDocsAsNumpyArraysTFidfTest = allDocsAsNumpyArraysTest.map(lambda x:
                                        (x[0], 1 if 'AU' in x[0] else 0, np.multiply(x[1], idfArrayTest)))
allDocsAsNumpyArraysTFidfTest.cache()

def calculate_f1(model, test_data):
    ytest_ypred = test_data.map(lambda x: (x[0], x[1], 1 if np.dot(model, x[2]) > 0 else 0))

    tp = ytest_ypred.filter(lambda x: x[1] == 1 and x[2] == 1).count()
    fp = ytest_ypred.filter(lambda x: x[1] == 0 and x[2] == 1).count()
    fn = ytest_ypred.filter(lambda x: x[1] == 1 and x[2] == 0).count()
    f1 = tp / (tp + 0.5 * (fp + fn))
    print('F1 score:', round(f1 * 100, 2), '%')

    fp_IDs = ytest_ypred.filter(lambda x: x[1] == 0 and x[2] == 1).map(lambda x: x[0]).take(3)
    print('IDs for False Positives:', fp_IDs)

calculate_f1(weights, allDocsAsNumpyArraysTFidfTest)

sc.stop()