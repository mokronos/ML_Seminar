import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
from utils import gen_dataset

path = "./data/"
df = gen_dataset("./data")
df = df.sample(frac=1)

split = 140
x_train = df['raw audio'][:split].values
y_train = df['rpm'][:split].values
x_test = df['raw audio'][split:].values
y_test = df['rpm'][split:].values

print(x_train)
print(y_train[1])



# input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])

# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create neural network   
NN = NeuralNetwork(x_train,y_train)
# train neural network
NN.train()

# create two new examples to predict                                   
#example = np.array([[1, 1, 0]])
#example_2 = np.array([[0, 1, 1]])

example = [x_test[0]]
example_2 =[x_test[1]]
 

# print the predictions for both examples                                   
#print(NN.predict(example), ' - Correct: ', example[0][0])
#print(NN.predict(example_2), ' - Correct: ', example_2[0][0])

print(NN.predict(example))
print(f"correct {y_test[0]}")
print(NN.predict(example_2))
print(f"correct {y_test[1]}")

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
