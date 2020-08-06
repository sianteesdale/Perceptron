#####

Student ID - 201069757

This folder contains a .py file which contains a perceptron algorithm for binary and multi-class classification. In order to run the code, open the file within a compatible software (e.g. Spyder or PyCharm).

#####

Running the code:
Please change the path on Line 8 to your directory which contains both the train.data and test.data files. 

#####

Binary Perceptron:
The function to run the binary perceptron is defined between lines 69-144. In order to run the code, uncomment the code on line 282 (to train the data) and line 286 (to test the trained model). 

The classes that can be used are: classes 1 and 2, classes 2 and 3, and classes 1 and 3. These are formatted as followed: class1_2 for the training data, and class1_2_t for the testing data. Please alter the data used in lines 282 and 286 as appropiate, taking care to add _t for test data on line 286. 

Do not alter any other parameters within lines 282 and 286.

Q5 - In order to check the most discriminative weight within class 1 and 2, uncomment the code on line 289 to print the weights, and define the data in lines 282 and 286 as class1_2 and class1_2_t respectively. The most discriminative will be the weight with the biggest value.

#####

Multi-Class Perceptron:
The function to run the binary perceptron is defined between lines 160-261. In order to run the code, uncomment the code on line 306 (to train the data) and line 310 (to test the trained model). 

The code on lines 306 and 310 does not need to be altered.

If the user wishes to add l2 regularisation to the multi-class perceptron (to stop overfitting), please comment out line 228, and uncomment line 229. 
To define the coefficient needed for l2 regularisation, please uncomment the coefficient on line 169, and define using either 0.01, 0.1, 1.0, 10.0 or 100.0 each time you run the model.

#####

