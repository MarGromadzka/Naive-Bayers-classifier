import numpy as np
import statistics as st
from math import sqrt, pi, exp, floor
from copy import deepcopy

def get_data():
    data = []
    
    with open("wine.data", "r") as file:
        for line in file.readlines():
            splitted_line = line.split(",")
            if len(splitted_line) == 14:
                data.append([list(map(float, splitted_line[1:14])), splitted_line[0]])
    return data

def choose_atributes(data, first, second):
    """" zwraca dane w postaci [[pierwszy_atrybut, drugi_atrybut], klasa]"""
    new_data = []
    for row in data:
        new_data.append([[row[0][first], row[0][second]], row[1]])
    return new_data

def divide_data(data, test_ratio, train_ratio):
    if (test_ratio + train_ratio != 1):
        print("Wrong test_ratio and train_ratio")
        return
    np.random.shuffle(data)
    train_size = int(train_ratio*len(data))
    train_set = data[0:train_size - 1]
    test_set = data[train_size:len(data)-1]
    return train_set, test_set

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

def calculate_prior_prob(data, classes):
    prior = []
    for given_class in classes:
        counter = 0
        for row in data:
            if given_class == row[1]:
                counter += 1
        prior.append(counter/len(data))
    return prior

def get_mean(data):
    attribute1 = []
    attribute2 = []
    for row in data:
        attribute1.append(row[0][0])
        attribute2.append(row[0][1])
    mean1 = np.mean(attribute1)
    mean2 = np.mean(attribute2)
    return mean1, mean2

def get_variance(data):
    attribute1 = []
    attribute2 = []
    for row in data:
        attribute1.append(row[0][0])
        attribute2.append(row[0][1])
    var1 = st.variance(attribute1) 
    var2 = st.variance(attribute2) 
    return var1, var2


def get_likelihood(x, mean, variance):
    if variance == 0:
            variance == 0.00000000001
    return (1/sqrt(2*pi*variance)) * exp((-((x - mean)**2))/(2*variance))

def train(data):
    class1, class2, class3 = classify_data(data)
    prior = calculate_prior_prob(data, ['1','2','3'])
    means1 = get_mean(class1)
    means2 = get_mean(class2)
    means3 = get_mean(class3)
    var1 = get_variance(class1)
    var2 = get_variance(class2)
    var3 = get_variance(class3)
    return prior, means1, means2, means3, var1, var2, var3

def get_posterior(x, prior, means, var):
    return prior * get_likelihood(x[0][0], means[0], var[0]) * get_likelihood(x[0][1], means[1], var[1])

def predict(x, prior, means1, means2, means3, var1, var2, var3):
    posterior1 = get_posterior(x, prior[0], means1, var1)
    posterior2 = get_posterior(x, prior[1], means2, var2)
    posterior3 = get_posterior(x, prior[2], means3, var3)
    max_p = max(posterior1, posterior2, posterior3)
    if max_p == posterior1:
        return '1'
    if max_p == posterior2:
        return '2'
    return '3'

def test(test_data, train_data):
    class1, class2, class3 = classify_data(test_data)
    prior, means1, means2, means3, var1, var2, var3 = train(train_data)
    error1 = 0
    error2 = 0
    error3 = 0
    for x in class1:
        if predict(x, prior, means1, means2, means3, var1, var2, var3) != '1':
            error1 += 1
    for x in class2:
        if predict(x, prior, means1, means2, means3, var1, var2, var3) != '2':
            error2 += 1
    for x in class3:
        if predict(x, prior, means1, means2, means3, var1, var2, var3) != '3':
            error3 += 1
    return 1-(error1/len(class1)), 1-(error2/len(class2)), 1-(error3/len(class3)), 1-((error1+error2+error3)/len(test_data))

def validation(train_set, n):
    set_size = floor(len(train_set)/n)
    k = 0
    sets = []
    best_accuracy = 0
    best_atributes = []
    while k < n:
        sets.append(train_set[k*set_size:(k+1)*set_size-1])
        k+=1
    with open("result.txt", 'w') as file:
        for i in range(0, 13):
            for k in range(i+1, 13):
                accuracies1 = []
                accuracies2 = []
                accuracies3 = []
                accuracies = []
                for j in range(0, len(sets)):
                    training_set = []
                    for l in range(0, len(sets)):
                        if l != j:
                            training_set += sets[l]
                    testing_set = choose_atributes(sets[j], i, k)
                    acc1, acc2, acc3, acc = test(testing_set, choose_atributes(training_set, i, k))
                    accuracies1.append(acc1)
                    accuracies2.append(acc2)
                    accuracies3.append(acc3)
                    accuracies.append(acc)
                file.write(f"Attributes: {i}, {k}\tAccuracy1: {round(np.mean(accuracies1),2)}\tAccuracy2: {round(np.mean(accuracies2),2)}\tAccuracy3: {round(np.mean(accuracies3),2)}\tAccuracy combined: {round(np.mean(accuracies),2)}\n")
                if best_accuracy < np.mean(accuracies):
                    best_accuracy = np.mean(accuracies)
                    best_atributes = [i, k]
    return best_atributes


def main():
    train_set, test_set = divide_data(get_data(), 0.2, 0.8)
    best_atributes = validation(train_set, 6)
    
    print(f"Attributes: {best_atributes[0]}, {best_atributes[1]}")
    accuracy1, accuracy2, accuracy3, accuracy = test(choose_atributes(test_set, best_atributes[0], best_atributes[1]), choose_atributes(train_set, best_atributes[0], best_atributes[1]))
    print(f"Class 1 accuracy: {accuracy1*100:.2f}%\nClass 2 accuracy: {accuracy2*100:.2f}%\nClass 3 accuracy: {accuracy3*100:.2f}%\nAccuracy: {accuracy*100:.2f}%")
    
main()