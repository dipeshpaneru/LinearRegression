import csv
import random
from decimal import Decimal


columnNumber = {"mean_radius": 2,
                "mean_texture": 3,
                 "mean_perimeter": 4, 
                 "mean_area": 5,
                 "mean_smoothness": 6,
                 "mean_symmetry": 10,
                 "mean_fractal_dimension": 11,
                 "worst_radius": 22,
                 "worst_area": 25,
                 "worst_symmetry": 30,
                 "lymph_node_status": 32}

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
            output = [Decimal(x[1]) for x in data]
            return output

    def performLinearRegression(self):
        self.iterate()

    def computePredictedOutput(self, thetas, data):
        predictedOutputs = []

        for x in self.getFeatures2D(data):
            y = Decimal(thetas[0])

            for idx, i in enumerate(x):
                y += (Decimal(thetas[idx + 1]) * Decimal(i))
            
            predictedOutputs.append(y)

        return predictedOutputs
    
    def calculateCost(self, predictedOutputs, data):
        outputs = self.getOutputs(data)
        featureLen = len(self.getFeatures2D(data))

        total = 0

        for idx, x in enumerate(predictedOutputs):
            result = (x - outputs[idx]) ** 2
            total += result

        cost = total / Decimal((2 * featureLen))
        return cost
    

    def calculateGradientDescent(self, B, index,  predictedOutputs, isIntercept):
        features = self.getFeatures2D(self.trainingData)
        outputs = self.getOutputs(self.trainingData)

        featureLen = Decimal(len(features))
        
        total = 0

        for idx, x in enumerate(features):

            if isIntercept:
                result = (predictedOutputs[idx] - outputs[idx])
            else:
                result = (predictedOutputs[idx] - outputs[idx]) * Decimal(x[index])

            total += result
        
        change = (Decimal(self.lr) * total) / featureLen

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
        print("The values of the Thetas/ Weights are")
        print(self.thetas)

        print("Mean Square Error of the Training data set is")
        predictedOutputs = self.computePredictedOutput(self.thetas, self.trainingData)
        cost = self.calculateCost(predictedOutputs, self.trainingData)
        print(cost)

        self.calculateMeanSqErr()


    def calculateMeanSqErr(self):
        print("Mean Square Error of the Test data set is")
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

        colsNo = self.getColNoArr(cols)

        finalData = []
        for x in data:
            interData = []

            for i in colsNo:
                interData.append(x[0][i])
            
            finalData.append((interData, x[1]))

        return finalData
    
    def performMinMaxScaling(self, cols, data):
        colsNo = self.getColNoArr(cols)

        for col in colsNo:
            MAX, MIN = self.getMaxAndMinOfCol(col, data)
            
            for x in data:
                x[0][col] = (Decimal(x[0][col]) - MIN) / (MAX - MIN)

        return data

    def getMaxAndMinOfCol(self, col, data):    
        dataOfCol = []

        for x in data:
            dataOfCol.append(Decimal(x[0][col]))

        return max(dataOfCol), min(dataOfCol)
    
    def getColNoArr(self, cols):
        colsNo = []

        for x in cols:
            colsNo.append(columnNumber[x])

        return colsNo
        





# Actually passing data to the regression model
dataEx = DataExtraction()
fileData = dataEx.getDataFromCsv("Cancer_dataset.csv") 
cleanData = dataEx.cleanUpData(fileData)        # Remove empty rows from dataset

TRAINING_DATA, TEST_DATA = dataEx.divideData(cleanData)


print("\n---------------- Question 1 -------------")

trainingData = dataEx.getReqColsFromData(['mean_texture'] ,TRAINING_DATA)
testData = dataEx.getReqColsFromData(['mean_texture'], TEST_DATA)


lr2 = LinearRegression(trainingData, testData, 50, [0, 0], 0.001) # passing in Training data, Test data, max no of iteration, starting theta values,
                                                                  # and the learning date
lr2.performLinearRegression()

print("\n---------------- Question 2 -------------")

trainingData2 = dataEx.getReqColsFromData(['mean_texture', 'lymph_node_status'] ,TRAINING_DATA)
testData2 = dataEx.getReqColsFromData(['mean_texture', 'lymph_node_status'], TEST_DATA)

lr2 = LinearRegression(trainingData2, testData2, 50, [0, 0, 0], 0.001)
lr2.performLinearRegression()


print("\n---------------- Question 3A -------------")
print("\nStep 1")

# Starting with mean_symmetry, chose it as a random parameter

trainingData3 = dataEx.getReqColsFromData(["mean_symmetry"] ,TRAINING_DATA)
testData3 = dataEx.getReqColsFromData(["mean_symmetry"], TEST_DATA)

lr3 = LinearRegression(trainingData3, testData3, 50, [0, 0], 0.001)
lr3.performLinearRegression()

print("\nStep 2")

# Adding lymph_node_status because it seemed to reduce overall cost in previous question

trainingData4 = dataEx.getReqColsFromData(["mean_symmetry", 'lymph_node_status'] ,TRAINING_DATA)
testData4 = dataEx.getReqColsFromData(["mean_symmetry", 'lymph_node_status'], TEST_DATA)

lr4 = LinearRegression(trainingData4, testData4, 50, [0, 0, 0], 0.001)
lr4.performLinearRegression()

print("\nStep 3")

# Adding mean_radius because it had a very low cost when model was built using it.

trainingData5 = dataEx.getReqColsFromData(["mean_radius", "mean_symmetry", 'lymph_node_status'] ,TRAINING_DATA)
testData5 = dataEx.getReqColsFromData(["mean_radius", "mean_symmetry", 'lymph_node_status'], TEST_DATA)

lr5 = LinearRegression(trainingData5, trainingData5, 50, [0, 0, 0, 0], 0.001)
lr5.performLinearRegression()

print("\nStep 4")

# Adding mean_smoothness as it does not increase cost.

trainingData6 = dataEx.getReqColsFromData(["mean_radius", "mean_symmetry", 'mean_smoothness', 'lymph_node_status'] ,TRAINING_DATA)
testData6 = dataEx.getReqColsFromData(["mean_radius", "mean_symmetry", 'mean_smoothness', 'lymph_node_status'], TEST_DATA)

lr6 = LinearRegression(trainingData6, trainingData6, 50, [0, 0, 0, 0, 0], 0.001)
lr6.performLinearRegression()

print("\nStep 5")

# Adding mean_fractal_dimension as it does not increase cost.

trainingData7 = dataEx.getReqColsFromData(["mean_radius",
                                            "mean_symmetry", 
                                            'mean_smoothness', 
                                            'lymph_node_status', 
                                            'mean_fractal_dimension'] ,TRAINING_DATA)
testData7 = dataEx.getReqColsFromData(["mean_radius", 
                                       "mean_symmetry", 
                                       'mean_smoothness', 
                                       'lymph_node_status', 
                                       'mean_fractal_dimension'], TEST_DATA)

lr7 = LinearRegression(trainingData7, testData7, 50, [0, 0, 0, 0, 0, 0], 0.001)
lr7.performLinearRegression()



print("\n---------------- Question 3B -------------")

print("\nStep 1")

# Using all 10 parameters, we see that cost is extremely large, some features being used is not
# usable without scaling or some sort of data manipulation.

trainingData8 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_perimeter",
                                            "mean_area", 
                                            "mean_smoothness", 
                                            "mean_symmetry", 
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_area",
                                            "worst_symmetry",
                                            "lymph_node_status"] ,TRAINING_DATA)

testData8 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_perimeter",
                                            "mean_area", 
                                            "mean_smoothness", 
                                            "mean_symmetry", 
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_area",
                                            "worst_symmetry",
                                            "lymph_node_status"], TEST_DATA)

lr8 = LinearRegression(trainingData8, testData8, 50, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.001)
lr8.performLinearRegression()

print("\nStep 2")
# We can see that the value that we got is very large so I will replace one by one 
# each value that is resulting such a huge number
# we have removed worst_area column in this iteration

trainingData9 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_perimeter",
                                            "mean_area", 
                                            "mean_smoothness", 
                                            "mean_symmetry", 
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_symmetry",
                                            "lymph_node_status"] ,TRAINING_DATA)
testData9 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_perimeter",
                                            "mean_area", 
                                            "mean_smoothness", 
                                            "mean_symmetry", 
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_symmetry",
                                            "lymph_node_status"], TEST_DATA)

lr9 = LinearRegression(trainingData9, testData9, 50, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.001)
lr9.performLinearRegression()

print("\nStep 3")

# Using same logic as before iteration I am removing mean_perimeter which is causing
# large cost

trainingData10 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_area", 
                                            'mean_smoothness',
                                            "mean_symmetry", 
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_symmetry",
                                            "lymph_node_status"] ,TRAINING_DATA)

testData10 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_area", 
                                            'mean_smoothness',
                                            "mean_symmetry", 
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_symmetry",
                                            "lymph_node_status"], TEST_DATA)

lr10 = LinearRegression(trainingData10, testData10, 50, [0, 0, 0, 0, 0, 0, 0, 0, 0], 0.001)
lr10.performLinearRegression()

print("\nStep 4")

# Using same logic as before iteration I am removing mean_area which is causing
# large cost

trainingData11 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_symmetry", 
                                             "mean_smoothness",
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_symmetry",
                                            "lymph_node_status"] ,TRAINING_DATA)
testData11 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_symmetry", 
                                            "mean_smoothness",
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "worst_symmetry",
                                            "lymph_node_status"], TEST_DATA)

lr11 = LinearRegression(trainingData11, testData11, 50, [0, 0, 0, 0, 0, 0, 0, 0], 0.001)
lr11.performLinearRegression()

print("\nStep 5")

# Removing worst_symmetry in this iteration

trainingData12 = dataEx.getReqColsFromData(['mean_radius',
                                            "mean_smoothness",
                                            "mean_symmetry", 
                                            "mean_fractal_dimension", 
                                            "worst_radius",
                                            "lymph_node_status"] ,TRAINING_DATA)
testData12 = dataEx.getReqColsFromData(['mean_radius', 
                                        "mean_smoothness",
                                        "mean_symmetry", 
                                        "mean_fractal_dimension", 
                                        "worst_radius",
                                        "lymph_node_status"], TEST_DATA)

lr12 = LinearRegression(trainingData12, testData12, 50, [0, 0, 0, 0, 0, 0, 0], 0.001)
lr12.performLinearRegression()  


print("\n-------------Question 4---------")
print("------------------4B ---------")

# Performing Min Max scaling on the model we received from Backward Stepwise regression,
# because it performed the best out of the previous ones

tempTrainingData = dataEx.performMinMaxScaling(['mean_radius', 
                                        "mean_smoothness",
                                        "mean_symmetry", 
                                        "mean_fractal_dimension", 
                                        "worst_radius",
                                        "lymph_node_status"],TRAINING_DATA)
tempTestData = dataEx.performMinMaxScaling(['mean_radius', 
                                        "mean_smoothness",
                                        "mean_symmetry", 
                                        "mean_fractal_dimension", 
                                        "worst_radius",
                                        "lymph_node_status"] ,TEST_DATA)

trainingData13 = dataEx.getReqColsFromData(['mean_radius', 
                                        "mean_smoothness",
                                        "mean_symmetry", 
                                        "mean_fractal_dimension", 
                                        "worst_radius",
                                        "lymph_node_status"] ,tempTrainingData)
testData13 = dataEx.getReqColsFromData(['mean_radius', 
                                        "mean_smoothness",
                                        "mean_symmetry", 
                                        "mean_fractal_dimension", 
                                        "worst_radius",
                                        "lymph_node_status"], tempTestData)


lr13 = LinearRegression(trainingData13, testData13, 50, [0, 0, 0, 0, 0, 0, 0], 0.001)
lr13.performLinearRegression()