# Building_Neural_network




1 Question

1.  Train and validate your own n-layer Neural Network on the Apparel dataset to predict the class label of a given apparel. You are free to choose the hyper-parameters, training strategy to handle large number of training data (Hint: Batch Size) architecture - number of hidden layers, number of nodes in each hidden layer etc. 

• Tasks: 

? Play with your neural network to report the best performing architecture for the given task. Also, include the activation function, loss function used in the report. 

? Contrast the e?ect of using di?erent activation functions on the validation data. Use ReLU, Sigmoid & Tanh as activation functions in the hidden layers.

? Report how the number of layers a?ects the performances of the model. Plot number of layers vs prediction loss graph. 

? Show how error is dependent on the number of epochs with appropriate plots. 

• Instructions: 

? Don’t use any library to code the Neural Network. Libraries can be used in handling the datasets, data pre-processing, reporting the performance measures, visualization. 

? Your neural network should take the architectural details (number of nodes in each layer, activation function to be used in each layer - ReLU/Sigmoid/Tanh) as inputs. 

? For simplicity, use the same activation function in all the hidden layers. 

? Test ?le will be shared later. You also need to submit the predicted labels of the test data. Instructions regarding test data are listed on the submission.txt ?le in the Apparel dataset folder. The test ?le would be in the same format as the training data but without the class label column. Please adhere to the instruction as there would be an automated checking to test the performance of your model.
 
? Submit both the code (with .py ?le) and the prediction CSV. 

? Save the weights of your best model to test the network on test data. [Hint: Dump the matrices after training and load them while testing. You can use numpy.savetext() for numpy arrays or pickle for dictionaries]
 
? Submit the code and prediction ?le only on moodle. Keep the weights with you, don’t submit them. 
? As a suggestion, take a small chunk of the dataset to test your implementation while coding the neural network. After the model is ready to run on the small chunk, use the whole dataset to ?x the best hyper-parameters. 


• Apparel Dataset: The dataset contains images of several classes of apparels. The description of the dataset is available in the README ?le provided. Please download the dataset from http://preon.iiit.ac.in/~ranajit_saha/files/ Apparel.zip

• Resources: ? UFLDL Tutorial ? Blog on Back-Propagation is very simple

2 Question
2 Consider the House Price Prediction dataset (Dataset Link). Suppose you need to predict the Sale Price of a house and for the task you want to use a neural network with 3 hidden layers. Write a report on how you would modify your above neural network for such task with proper reasoning.