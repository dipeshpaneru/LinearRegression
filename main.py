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



data = getDataFromCsv("Cancer_dataset.csv", 4, 33)
data.pop(0)


