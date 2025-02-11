import csv
import random

# Linear regression model Class with all the components required for linear regression
class LinearRegression:
    iterationCount = 0
    learnedB0 = 0
    learnedB1 = 0
    trainingData = None
    testData = None

    def __init__(self, DATA, maxNoOfIterations, B0, B1, lr):
        self.DATA = DATA
        self.maxNoOfIterations = maxNoOfIterations
        self.B0 = B0
        self.B1 = B1
        self.lr = lr

    def divideData(self):
        dataLen = len((self.DATA))
        trainingCount = int((dataLen / 0.8))

        randomData = random.choice(self.DATA)
        randomTrainingData = random.sample(randomData, trainingCount)
        testData = randomData - randomTrainingData

        self.trainingData = randomTrainingData
        self.testData = testData
        
        
    def getFeatures(self):
        features = [float(x[0]) for x in self.DATA]
        return features

    def getOutputs(self):
        output = [float(x[1]) for x in self.DATA]
        return output

    def performLinearRegression(self):
        self.iterate(self.B0, self.B1)

    def computePredictedOutput(self, B0, B1):
        predictedOutputs = []

        for x in self.getFeatures():
            y = B0 + (B1 * x)
            predictedOutputs.append(y)

        return predictedOutputs
    
    def calculateCost(self, predictedOutputs):
        outputs = self.getOutputs()
        featureLen = len(self.getFeatures())

        total = 0

        for idx, x in enumerate(predictedOutputs):
            result = (x - outputs[idx]) ** 2
            total += result

        cost = total / float((2 * featureLen))
        return cost
    

    def calculateGradientDescent(self, B, predictedOutputs, isIntercept):
        features = self.getFeatures()
        outputs = self.getOutputs()

        featureLen = float(len(features))
        
        total = 0

        for idx, x in enumerate(features):

            if isIntercept:
                result = (predictedOutputs[idx] - outputs[idx])
            else:
                result = (predictedOutputs[idx] - outputs[idx]) * x

            total += result
        
        change = (self.lr * total) / featureLen

        newB = B - change
        return newB
    
    def iterate(self, B0, B1):
        self.iterationCount = self.iterationCount + 1

        predictedOutputs = self.computePredictedOutput(B0, B1)
        outputs = self.getOutputs()
        
        cost = self.calculateCost(predictedOutputs)
        
        newB0 = self.calculateGradientDescent(B0, predictedOutputs, True)
        newB1 = self.calculateGradientDescent(B1, predictedOutputs, False)

        if self.iterationCount < self.maxNoOfIterations:
            self.iterate(newB0, newB1)
        else:
            self.learnedB0 = newB0
            self.learnedB1 = newB1
            self.printLearnedWeights()

    
    def printLearnedWeights(self):
        print(self.learnedB0)
        print(self.learnedB1)

# Getting and cleaning up data before passing it through the regression model

def getDataFromCsv(file_path, column1_index, column2_index):
    
    data = []
    with open(file_path, 'r') as file:
        
        reader = csv.reader(file)
        for row in reader:
            try:
                data.append((row[column1_index], row[column2_index]))
            except:
                print(f"ERROR: Error while accessing data from CSV file")
    return data

def cleanUpData(data):
    data.pop(0)

    for x in data:
        if x[0] == "":
            data.remove(x)
            
    return data


tempData = getDataFromCsv("Cancer_dataset.csv", 4, 33)
data = cleanUpData(tempData)

# Actually passing data to the regression model
regressionModel = LinearRegression(data, 100, 0.2, 0.2, 0.01)
regressionModel.performLinearRegression()


