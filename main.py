import csv
import random

# Linear regression model Class with all the components required for linear regression
class LinearRegression:
    
    trainingData = None
    testData = None

    def __init__(self, trainingData, testData, maxNoOfIterations, thetas, lr):
        self.trainingData = trainingData
        self.testData = testData
        self.maxNoOfIterations = maxNoOfIterations
        self.thetas = thetas
        self.lr = lr
        
        
    def getFeatures2D(self, data):
        if data != None:
            features = [x[0] for x in data]
            return features

    def getOutputs(self, data):
        if data != None:
            output = [float(x[1]) for x in data]
            return output

    def performLinearRegression(self):
        self.iterate()

    def computePredictedOutput(self, thetas, data):
        predictedOutputs = []

        for x in self.getFeatures2D(data):
            y = thetas[0]

            for idx, i in enumerate(x):
                y += (float(thetas[idx + 1]) * float(i))
            
            predictedOutputs.append(y)

        return predictedOutputs
    
    def calculateCost(self, predictedOutputs, data):
        outputs = self.getOutputs(data)
        featureLen = len(self.getFeatures2D(data))

        total = 0

        for idx, x in enumerate(predictedOutputs):
            result = (x - outputs[idx]) ** 2
            total += result

        cost = total / float((2 * featureLen))
        return cost
    

    def calculateGradientDescent(self, B, index,  predictedOutputs, isIntercept):
        features = self.getFeatures2D(self.trainingData)
        outputs = self.getOutputs(self.trainingData)

        featureLen = float(len(features))
        
        total = 0

        for idx, x in enumerate(features):

            if isIntercept:
                result = (predictedOutputs[idx] - outputs[idx])
            else:
                result = (predictedOutputs[idx] - outputs[idx]) * float(x[index])

            total += result
        
        change = (self.lr * total) / featureLen

        newB = B - change
        return newB
    
    def iterate(self):

        for i in range(self.maxNoOfIterations):

            predictedOutputs = self.computePredictedOutput(self.thetas, self.trainingData)

            for idx, b in enumerate(self.thetas):
                if idx == 0:
                    newB = self.calculateGradientDescent(b, (idx - 1), predictedOutputs, True)
                else:
                    newB = self.calculateGradientDescent(b, (idx - 1), predictedOutputs, False)

                self.thetas[idx] = newB
        
        self.printResults()

    
    def printResults(self):
        print("The values of the Thetas are")
        print(self.thetas)

        print("Cost of the Training data set is")
        predictedOutputs = self.computePredictedOutput(self.thetas, self.trainingData)
        cost = self.calculateCost(predictedOutputs, self.trainingData)
        print(cost)

        self.calculateMeanSqErr()


    def calculateMeanSqErr(self):
        print("Cost of the Test data set is")
        predictedOutputs = self.computePredictedOutput(self.thetas, self.testData)
        cost = self.calculateCost(predictedOutputs, self.testData)
        print(cost)



# Getting and cleaning up data before passing it through the regression model

class DataExtraction:
    def getDataFromCsv(self, file_path):
        
        data = []
        with open(file_path, 'r') as file:
            
            reader = csv.reader(file)
            for row in reader:
                try:
                    parameters = []

                    for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34]:
                        parameters.append(row[x])

                    data.append((parameters, row[33]))
                except:
                    print(f"ERROR: Error while accessing data from CSV file")
        return data

    def cleanUpData(self, data):
        data.pop(0)
        tempArray = []

        for idx, x in enumerate(data):
            hasEmpty = False

            for i in x[0]:
                if i == "":
                    hasEmpty = True
            
            if not hasEmpty:
                tempArray.append(data[idx])
    
        return tempArray

    def divideData(self, data):
        dataLen = len((data))
        trainingCount = int((dataLen * 0.8))

        random.shuffle(data)

        randomTrainingData = random.sample(data, trainingCount)

        for x in randomTrainingData:
            data.remove(x)

        testData = data

        return randomTrainingData, testData

    def getReqColsFromData(self, cols, data):
        finalData = []
        for x in data:
            interData = []

            for i in cols:
                interData.append(x[0][i])
            
            finalData.append((interData, x[1]))

        return finalData




# Actually passing data to the regression model
dataEx = DataExtraction()
fileData = dataEx.getDataFromCsv("Cancer_dataset.csv")
cleanData = dataEx.cleanUpData(fileData)
TRAINING_DATA, TEST_DATA = dataEx.divideData(cleanData)


print("\n---------------- Question 1 -------------")

trainingData = dataEx.getReqColsFromData([3] ,TRAINING_DATA)
testData = dataEx.getReqColsFromData([3], TEST_DATA)


lr2 = LinearRegression(trainingData, testData, 1000, [0, 0], 0.001)
lr2.performLinearRegression()

# print("\n---------------- Question 2 -------------")

trainingData2 = dataEx.getReqColsFromData([3, 32] ,TRAINING_DATA)
testData2 = dataEx.getReqColsFromData([3, 32], TEST_DATA)

lr2 = LinearRegression(trainingData2, testData2, 1000, [0, 0, 0], 0.001)
lr2.performLinearRegression()


print("\n---------------- Question 3 -------------")
print("\nStep 1")
# # I have chosen lymph_node_status first because in question 2
# # adding lymph_node_status to the model seemed to better the model performance

trainingData3 = dataEx.getReqColsFromData([10] ,TRAINING_DATA)
testData3 = dataEx.getReqColsFromData([10], TEST_DATA)

lr3 = LinearRegression(trainingData3, testData3, 1000, [0, 0], 0.001)
lr3.performLinearRegression()

print("\nStep 2")
# # I have chosen mean_radius in the for second step,
# # because radius should impact the size 

trainingData4 = dataEx.getReqColsFromData([32, 10] ,TRAINING_DATA)
testData4 = dataEx.getReqColsFromData([32, 10], TEST_DATA)

lr4 = LinearRegression(trainingData4, testData4, 1000, [0, 0, 0], 0.001)
lr4.performLinearRegression()

print("\nStep 3")
#  # In third step I am adding mean_smoothness because mean_radius, 
#  # mean_parameter and mean_area seems to have some colinearity, thus I am choosing 
#  # something that is not these two

trainingData5 = dataEx.getReqColsFromData([2, 10, 32] ,TRAINING_DATA)
testData5 = dataEx.getReqColsFromData([2, 10, 32], TEST_DATA)

lr5 = LinearRegression(trainingData5, trainingData5, 1000, [0, 0, 0, 0], 0.001)
lr5.performLinearRegression()

print("\nStep 4")

trainingData6 = dataEx.getReqColsFromData([2, 6, 10, 32] ,TRAINING_DATA)
testData6 = dataEx.getReqColsFromData([2, 6, 10, 32], TEST_DATA)

lr6 = LinearRegression(trainingData6, trainingData6, 1000, [0, 0, 0, 0, 0], 0.001)
lr6.performLinearRegression()

print("\nStep 5")

trainingData7 = dataEx.getReqColsFromData([2, 6, 10, 32, 11] ,TRAINING_DATA)
testData7 = dataEx.getReqColsFromData([2, 6, 10, 32, 11], TEST_DATA)

lr7 = LinearRegression(trainingData7, testData7, 1000, [0, 0, 0, 0, 0, 0], 0.001)
lr7.performLinearRegression()

# print("\nStep 6")

# trainingData8 = dataEx.getReqColsFromData([2, 22, 30, 32, 11, 6] ,TRAINING_DATA)
# testData8 = dataEx.getReqColsFromData([2, 22, 30, 32, 11, 6], TEST_DATA)

# lr8 = LinearRegression(trainingData8, testData8, 1000, [0, 0, 0, 0, 0, 0, 0], 0.001)
# lr8.performLinearRegression()

# print("\nStep 7")

# trainingData8 = dataEx.getReqColsFromData([2, 5, 30, 32, 11, 6, 22] ,TRAINING_DATA)
# testData8 = dataEx.getReqColsFromData([2, 5, 30, 32, 11, 6, 22], TEST_DATA)

# lr8 = LinearRegression(trainingData8, testData8, 1000, [0, 0, 0, 0, 0, 0, 0, 0], 0.001)
# lr8.performLinearRegression()