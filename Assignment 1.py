#Import the relevant libraries
import numpy as np
import csv
import random

## Train data
# Loading in data using csv and np
path = 'F:/Data Mining/Assignment 1/data'
train = open(path + '/train.data', 'rt')
reader = csv.reader(train, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
train_data = np.array(x)

#Create empty lists for the pairs of classes
class1_2 = []
class2_3 = []
class1_3 = []

#Separate data into pairs of classes
for row in train_data:
    if row[4] == 'class-1':
        #Append the row to the relevant pairs
        class1_2.append(row)
        class1_3.append(row)
    if row[4] == 'class-2':
        class2_3.append(row)
        class1_2.append(row)
    if row[4] == 'class-3':
        class1_3.append(row)
        class2_3.append(row)

## Test data
# Repeat the same steps are previously but with test.data        
test = open(path + '/test.data', 'rt')
reader = csv.reader(test, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
test_data = np.array(x)

class1_2_t = []
class2_3_t = []
class1_3_t = []

for row in test_data :
    if row[4] == 'class-1':
        class1_2_t.append(row)
        class1_3_t.append(row)
    if row[4] == 'class-2':
        class2_3_t.append(row)
        class1_2_t.append(row)
    if row[4] == 'class-3':
        class1_3_t.append(row)
        class2_3_t.append(row)
        

##############################################################################

"""
The Binary Perceptron function assigns the first class read in as -1, and the
second class as 1.

This function will take in a pair of classes in a list, weights, bias and a 
string defining whether the data is "Train" or "Test".

It will either produce weights and a bias (if defined as a train data), or 
apply the weights of the train data to the test data to predict the 
classification of each instance.
"""
        
def PerceptronBinary(data, w, b,trainOrTest):
    #Split the data into two, with the features in x, and class' in y
    data = np.hsplit((np.array(data)),
                     np.array([4, 8]))
    x = data[0].astype(float)
    y = np.array(np.unique(data[1], return_inverse=True))
    #Retain the names of the classes for printing later
    name1 = y[0][0]
    name2 = y[0][1]
    y = np.array(y[1])
    #Convert the 0 in y to -1
    y[y < 1] = -1
    
    #Create variables for pocket algorithm
    bestW = w
    bestB = b
    bestAcc = 0
    
    #If function is defined as test, run 1 iteration 
    if trainOrTest != "Train":
        num_iterations = 1
    #If function is defined as train, run 20 iterations
    else:
        num_iterations = 20
        #Create variables for weights and bias
        w = [0.0, 0.0, 0.0, 0.0]
        b = 0
    
    #For the number of iterations
    for epoch in range(num_iterations):
        #Change accuracy to 0
        acc = 0
        #Join together the x and y and shuffle the data
        zipedList = list(zip(x, y))
        random.shuffle(zipedList)
        x, y = zip(*zipedList)
        
        #For each row in x, set activation to 0
        for i in range(len(x)):
            a = 0
            #For each feature in each row, calculate the activation
            for j in range(len(x[i])):
                a += (w[j] * x[i][j]) + b
            #If the a > 0, adjust to 1, if a < 0 then change to -1
            if a > 0 :
                a = 1
            else :
                a = -1
            #If the activation * the classification is <= 0 then update 
            # weights and bias on train dataset; otherwise, increase accuracy 
            # score by 1.
            if (a * y[i]) <= 0:
                if trainOrTest == "Train":
                    for j in range(len(w)):
                        w[j] = w[j] + (y[i] * x[i][j])
                    b += y[i]
            else:
                acc += 1
        #If the accuracy recorded is greater than the bestAccuracy recorded,
        # then update the bestAcc, and the weights and bias if train data
        if bestAcc < acc:
            bestAcc = acc
            if trainOrTest == "Train":
                bestW = w.copy()
                bestB = b
                
    #Print the model accuracies for train and test models
    print(trainOrTest,"model accuracy for", name1, "/",name2+":", ((bestAcc) / len(x)) * 100, "%")
    #Print how many lines were correct
    print("\tGot: ", (bestAcc), "/", len(y), "lines correct\n") 
    
    #If the data was training data, then return the bestWeights and bestBias
    if trainOrTest == "Train":
        return bestW, bestB
    else:
        return 
 
##############################################################################

"""
The Multi-Class Perceptron function utilises the 1-vs-rest algorithm, in which
the class of interest is given a 1, and the other classes are assigned -1.

This function will take in a whole dataset with three classes, weights and 
bias in an array, and a string defining whether the data is "Train" or "Test".

It will either produce an array of weights and an array of bias values (if 
defined as a train data), or apply the weights and bias of the train data to 
the test data to predict the classification of each instance.
"""
    
def PerceptronMultiClass(data,wArray,bArray,trainOrTest):
    #Split the data into two, with the features in x, and class' in y
    data = np.hsplit((np.array(data)),
                     np.array([4, 8]))
    x = data[0].astype(float)
    y = np.array(np.unique(data[1], return_inverse=True))
    y = np.array(y[1])
    
    #Define coefficient for l2 regularisation
    #coeff = 0.01

    #Create variables for pocket algorithm
    bestmultiW = []
    bestmultiB = []
    #Create a copy of y
    z = y.copy()

    #For the number of classes in dataset
    for i in range(3):
        #Reset bestAccuracy to 0
        bestAcc = 0
        #If data is train, reset the weights, bias, bestW, bestB and
        #set the number of iterations to 20
        if trainOrTest == "Train":
            w = [0.0, 0.0, 0.0, 0.0]
            b = 0
            bestW = []
            
            bestB = 0
            num_iterations = 20
        #If data is test, set the weight and bias to the relevant loop, and 
        #set the model iterations to 1
        else:
            w = wArray[i]
            b = bArray[i]
            num_iterations = 1
        
        #For the number of values in z
        for j in range(z.shape[0]):
            #If the number == 2, then change to 1, otherwise change to -1
            if z[j] == 2:
                y[j] = 1
            else:
                y[j] = -1
        #Add 1 to z for the next loop
        z += 1
        y = np.array(y)
         
        #For the number of iterations
        for epoch in range(num_iterations):
                #Change accuracy to 0
                acc = 0
                #Join together the x and y and shuffle the data
                zipedList = list(zip(x, y))
                random.shuffle(zipedList)
                x, y = zip(*zipedList)
                
                #For each row in x, set activation to 0
                for k in range(len(x)):
                    a = 0.0
                    #For each feature in each row, calculate the activation
                    for m in range(len(x[k])):
                        a += (w[m] * x[k][m]) + b
                    #If the activation * the classification is <= 0 then update 
                    # weights and bias on train dataset; otherwise, increase accuracy 
                    # score by 1.
                    if (a * y[k]) <= 0:
                        if trainOrTest == "Train":
                            for j in range(len(w)):
                                w[m] = w[m] + (y[k] * x[k][m])
                                #w[m] = w[m] + (y[k] * x[k][m]) - (2*coeff*w[m])
                            b += y[k]
                    else:
                        acc += 1
                #If the accuracy recorded is greater than the bestAccuracy recorded,
                # then update the bestAcc, and the weights and bias if train data        
                if bestAcc < acc:
                    bestAcc = acc
                    if trainOrTest == "Train":
                        bestW = w.copy()
                        bestB = b
        #Print the model accuracies for train and test models            
        print(trainOrTest,"model accuracy for Class", 3-i, ":", round((bestAcc/len(x) *100), 2), "%")
        #Print how many lines were correct
        print("\tGot:", (bestAcc), "/", len(y), "lines correct\n") 
            
        #If the data is train, append the bestWeights and bestBias of each
        # loop of the function to bestmultiW and bestmultiB
        if trainOrTest == "Train":
            bestmultiW.append(bestW)
            bestmultiB.append(bestB)
        
        #Reset x and y ready for the next loop of the function
        x = data[0].astype(float)
        y = np.array(np.unique(data[1], return_inverse=True))
        y = np.array(y[1])
    
    #If the data was training data, then return the bestWeights and bestBias
    if trainOrTest == "Train":
        return bestmultiW, bestmultiB
    else:
        return

##############################################################################

## Run the models ##
        
"""
Binary Perceptron

For the train model:
Change the data within the function to class1_2, class2_3 or class1_3
Keep the weights and bias as 0

For the test model:
Change the data within function to class1_2_t, class2_3_t or class1_3_t 
Change the weights and bias to w and b

When defining the train or test models, use "Train" or "Test"
"""        
#Train model
#Save the weights and bias from the train model
#w, b = PerceptronBinary(class1_2, 0, 0, "Train")

#Test model
#Uses weights and bias saved from train model
#PerceptronBinary(class1_2_t, w, b,"Test")

#Q5 - Print the weights of model for class1_2
#print(w)

"""
Multi-Class Perceptron

For the train model:
Keep the data as train_data
Keep the weights and bias as 0

For the test model:
Keep the data as test_data
Keep the weights and bias as wArray and bArray

When defining the train or test models, use "Train" or "Test"
"""  
#Train model
#Save the weights and bias from the train model
wArray, bArray = PerceptronMultiClass(train_data, 0, 0,"Train")

#Test model
#Uses weights and bias saved from train model
PerceptronMultiClass(test_data, wArray, bArray,"Test")





