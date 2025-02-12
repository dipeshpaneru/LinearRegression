import csv
import random

# Linear regression model Class with all the components required for linear regression
class LinearRegression:
    
    th0 = 0
    th1 = 0
    trainingData = None
    testData = None

    def __init__(self, DATA, maxNoOfIterations, B0, B1, lr):
        self.DATA = DATA
        self.maxNoOfIterations = maxNoOfIterations
        self.B0 = B0
        self.B1 = B1
        self.lr = lr

        self.divideData()

    def divideData(self):
        dataLen = len((self.DATA))
        trainingCount = int((dataLen * 0.8))

        data = self.DATA
        random.shuffle(data)

        randomTrainingData = random.sample(data, trainingCount)
        testData = list(set(data) - set(randomTrainingData))

        self.trainingData = randomTrainingData
        self.testData = testData
        
        
    def getFeatures(self):
        if self.trainingData != None:
            features = [float(x[0]) for x in self.trainingData]
            return features

    def getOutputs(self):
        if self.trainingData != None:
            output = [float(x[1]) for x in self.trainingData]
            return output

    def performLinearRegression(self):
        self.iterate()

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
    
    def iterate(self):

        for i in range(self.maxNoOfIterations):

            predictedOutputs = self.computePredictedOutput(self.th0, self.th1)
            
            newB0 = self.calculateGradientDescent(self.th0, predictedOutputs, True)
            newB1 = self.calculateGradientDescent(self.th1, predictedOutputs, False)

            self.th0 = newB0
            self.th1 = newB1
        
        self.printResults()

    
    def printResults(self):
        print(self.th0)
        print(self.th1)
        predictedOutputs = self.computePredictedOutput(self.th0, self.th1)
        cost = self.calculateCost(predictedOutputs)
        print(cost)



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
    tempData = data

    for idx, x in enumerate(tempData):
        if x[0] == "":
            data.pop(idx)
            
    return data


tempData = getDataFromCsv("Cancer_dataset.csv", 4, 33)
data = cleanUpData(tempData)

# Actually passing data to the regression model
regressionModel = LinearRegression(data, 100, 0, 0, 0.001)
regressionModel.performLinearRegression()


