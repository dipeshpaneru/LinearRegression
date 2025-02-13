import csv
import random
import copy

# Linear regression model Class with all the components required for linear regression
class LinearRegression:
    
    trainingData = None
    testData = None

    def __init__(self, DATA, maxNoOfIterations, thetas, lr):
        self.DATA = DATA
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

def getDataFromCsv(file_path):
    
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

def cleanUpData(data):
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

def divideData(data):
    dataLen = len((data))
    trainingCount = int((dataLen * 0.8))

    random.shuffle(data)

    randomTrainingData = random.sample(data, trainingCount)

    for x in randomTrainingData:
        data.remove(x)

    testData = data

    return randomTrainingData, testData





# Actually passing data to the regression model

print("\n---------------- Question 1 -------------")

tempData = getDataFromCsv("Cancer_dataset.csv")
print(len(tempData))
data = cleanUpData(tempData)
print(len(data))
trainingData, testData = divideData(data)

print(len(trainingData), len(testData))

# lr1 = LinearRegression(data1, 1000, [0, 0], 0.001)
# lr1.divideData()
# lr1.performLinearRegression()

# print("\n---------------- Question 2 -------------")

# tempData2 = getDataFromCsv("Cancer_dataset.csv")
# data2 = cleanUpData(tempData2)









# lr2 = LinearRegression(data2, 1000, [0, 0, 0], 0.001)
# lr2.divideData()
# lr2.performLinearRegression()


# print("\n---------------- Question 3 -------------")
# print("Step 1")
# # I have chosen lymph_node_status first because in question 2
# # adding lymph_node_status to the model seemed to better the model performance

# tempData3 = getDataFromCsv("Cancer_dataset.csv", [34], 33)
# data3 = cleanUpData(tempData3)

# lr3 = LinearRegression(data3, 1000, [0, 0], 0.001)
# lr3.divideData()
# lr3.performLinearRegression()

# print("Step 2")
# # I have chosen mean_radius in the for second step,
# # because radius should impact the size 

# tempData3 = getDataFromCsv("Cancer_dataset.csv", [3, 34], 33)
# data3 = cleanUpData(tempData3)

# regressionModel = LinearRegression(data3, 1000, [0, 0, 0], 0.001)
# regressionModel.performLinearRegression()

# print("Step 3")
#  # In third step I am adding mean_smoothness because mean_radius, 
#  # mean_parameter and mean_area seems to have some colinearity, thus I am choosing 
#  # something that is not these two

# tempData3 = getDataFromCsv("Cancer_dataset.csv", [7, 3, 34], 33)
# data3 = cleanUpData(tempData3)

# regressionModel = LinearRegression(data3, 1000, [0, 0, 0, 0], 0.001)
# regressionModel.performLinearRegression()

# print("Step 4")

# tempData3 = getDataFromCsv("Cancer_dataset.csv", [7, 3, 10, 34], 33)
# data3 = cleanUpData(tempData3)

# regressionModel = LinearRegression(data3, 1000, [0, 0, 0, 0, 0], 0.001)
# regressionModel.performLinearRegression()

# # print("Step 5")

# tempData3 = getDataFromCsv("Cancer_dataset.csv", [7, 3, 10, 11, 34], 33)
# data3 = cleanUpData(tempData3)

# regressionModel = LinearRegression(data3, 1000, [0, 0, 0, 0 ,0, 0], 0.001)
# regressionModel.performLinearRegression()