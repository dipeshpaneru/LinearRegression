import csv

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


data = getDataFromCsv("Cancer_dataset.csv", 4, 33)
DATA = cleanUpData(data)

def performLinearRegression():

    B0 = 0.2
    B1 = 0.2

    iterate(B0, B1)


def getFeatures():
    features = [float(x[0]) for x in DATA]
    return features

def getOutputs():
    output = [float(x[1]) for x in DATA]
    return output


def computePredictedOutput(B0, B1):
    predictedOutputs = []

    for x in getFeatures():
        y = B0 + (B1 * x)
        predictedOutputs.append(y)

    return predictedOutputs

def calculateCost(predictedOutput, actualOutput):

    total = 0

    result = (predictedOutput - actualOutput) ** 2
    total += result

    cost = total / 2
    return cost


def iterate(B0, B1):
    predictedOutputs = computePredictedOutput(B0, B1)
    outputs = getOutputs()
    costs = []

    for y in predictedOutputs:
        cost = calculateCost(y, outputs.pop(0))
        costs.append(cost)

    print(getOutputs)
    print(costs)

   

performLinearRegression()



