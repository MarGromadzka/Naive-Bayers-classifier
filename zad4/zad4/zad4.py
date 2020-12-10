import numpy as np

def get_data():
    data = []
    
    with open("wine.data", "r") as file:
        for line in file.readlines():
            splitted_line = line.split(",")
            if len(splitted_line) == 14:
                data.append([list(map(float, splitted_line[1:13])), splitted_line[0]])
    return data

def choose_atributes(data, first, second):
    new_data = []
    for row in data:
        new_data.append([[row[0][first], row[0][second]], row[1]])
    return new_data

def divide_data(data, test_ratio, train_ratio):
    validation_ratio = 1 - test_ratio - train_ratio
    if (validation_ratio <= 0):
        print("Wrong test_ratio and train_ratio")
        return
    np.random.shuffle(data)
    train_size = int(train_ratio*len(data))
    test_size = int(test_ratio*len(data))
    train_set = data[:train_size - 1]
    test_set = data[train_size:train_size+test_size]
    validation_set = data[train_size+1+test_size:len(data)-1]
    return train_set, test_set, validation_set

def classify_data(data):
    class1 = []
    class2 = []
    class3 = []
    for row in data:
        if row[1] == '1':
            class1.append(row)
        elif row[1] == '2':
            class2.append(row)
        elif row[1] == '3':
            class3.append(row)
    return class1, class2, class3

data = choose_atributes(get_data(), 0, 1)
classify_data(data)
