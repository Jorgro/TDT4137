import numpy as np


class Perceptron:
    # can drop threshold?
    def __init__(self, no_inputs, learning_data, labels,
                 threshold=100,
                 learning_rate=0.05):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(2 + 1)  # +1 because of bias.
        self.learning_data = learning_data
        self.labels = labels
        self.no_inputs = no_inputs

    def predict(self, input):
        # 0 is bias

        y_p = np.dot(input, self.weights[1:]) + self.weights[0]
        if y_p > 0:
            return 1  # returns activation 1 if y is over 0
        return 0  # returns activation 0 if y is less than or equal to 0

    def train(self):
        t = 0
        # setting max 1000 iterations of training.
        while not self.check_convergence() and t < 996:
            i = t % len(self.learning_data)
            x = self.learning_data[i]

            error = self.labels[i] - self.predict(x)
            self.weights[1:] += self.learning_rate * x * error
            self.weights[0] += self.learning_rate * error

            t += 1

        print("Weights: ", self.weights)

    def check_convergence(self):
        # Go through all training input and check that prediction is correct.
        # If it is then we have converged!
        for data, label in zip(self.learning_data, self.labels):
            if self.predict(data) != label:
                return False
        return True


training = [
    np.array([1, 1]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([0, 0]),
]

labels = np.array([1, 0, 0, 0])

perceptron = Perceptron(2, training, labels)
perceptron.train()

print(perceptron.predict(np.array([1, 0])))
print(perceptron.predict(np.array([0, 1])))
print(perceptron.predict(np.array([0, 0])))
print(perceptron.predict(np.array([1, 1])))

training = [
    np.array([1, 1]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([0, 0]),
]

labels = np.array([1, 1, 1, 0])

perceptron = Perceptron(2, training, labels)
perceptron.train()

print(perceptron.predict(np.array([1, 0])))
print(perceptron.predict(np.array([0, 1])))
print(perceptron.predict(np.array([0, 0])))
print(perceptron.predict(np.array([1, 1])))

training = [
    np.array([1, 1]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([0, 0]),
]

labels = np.array([0, 1, 1, 0])

perceptron = Perceptron(2, training, labels)
perceptron.train()

print(perceptron.predict(np.array([1, 0])))
print(perceptron.predict(np.array([0, 1])))
print(perceptron.predict(np.array([0, 0])))
print(perceptron.predict(np.array([1, 1])))
