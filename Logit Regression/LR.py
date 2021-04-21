import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


ploty =[]
class LogisticRegression:

    def __init__(self, learning_rate=0.1, n_iterations=5001):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None
        
    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))
    @staticmethod
    def _binary_cross_entropy(y, y_hat):
        def safe_log(x): 
            return 0 if x == 0 else np.log(x)
        total = 0
        for curr_y, curr_y_hat in zip(y, y_hat):
            total += (curr_y * safe_log(curr_y_hat) + (1 - curr_y) * safe_log(1 - curr_y_hat))
        return - total / len(y)
        
    def fit(self, X, y):
        # 1. Initialize coefficients
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            probability = self._sigmoid(linear_pred)
            

            # Calculate derivatives
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (probability - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(probability - y))
            
            if i % 50 == 0:
            	test = [1 if j > 0.5 else 0 for j in probability]
            	print('Accuracy at iteration ' + str(i) + ': '+ str(accuracy(y, test, 1000)))
            	ploty.append(accuracy(y, test, 1000))


            
            # Update the coefficients
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

            
    def fit_sgd(self, X, y):
        # 1. Initialize coefficients
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # 2. Perform stochastic gradient descent
        for i in range(self.n_iterations):
        	index = random.choice(range(X.shape[0]))
        	if(i == 0):
        		X=X.to_numpy()
        	linear_pred = np.dot(X[index], self.weights) + self.bias
        	probability = self._sigmoid(linear_pred)
        	partial_w = (1 / X.shape[0]) * (2 * np.dot(X[index].T, (probability - y[index])))
        	partial_d = (1 / X.shape[0]) * (2 * np.sum(probability - y[index]))
        	if i % 50 == 0:
        		linear_pred = np.dot(X, self.weights) + self.bias
        		probability = self._sigmoid(linear_pred)
        		test = [1 if j > 0.5 else 0 for j in probability]
        		print('Accuracy at iteration ' + str(i) + ': '+ str(accuracy(y, test, 1000)))
        		ploty.append(accuracy(y, test, 1000))
        	self.weights -= self.learning_rate * partial_w
        	self.bias -= self.learning_rate * partial_d

         
    def predict_proba(self, X):
    	linear_pred = np.dot(X, self.weights) + self.bias
    	return self._sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):

        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities], probabilities

def accuracy(predicted, actual, number):
	values = 0.0
	for i in range(number):
		if(predicted[i] == actual[i]):
			values += 1
	return values*100/number
def truepositives(predicted, actual, number):
	value = 0
	for i in range(number):
		if(predicted[i] == 1 and actual[i] == 1):
			value += 1
	return value

def falsepositives(predicted, actual, number):
	value = 0
	for i in range(number):
		if(predicted[i] == 1 and actual[i]==0):
			value+=1
	return value

def falsenegatives(predicted, actual, number):
	value = 0
	for i in range(number):
		if(predicted[i] == 0 and actual[i] == 1):
			value+=1
	return value


print('Stochastic Gradient Descent')

data = pd.read_csv (r'dataset_LR.csv')
X_data = data[data.columns[0:4]]
y = data['class']
y = y.to_numpy()
model = LogisticRegression()
model.fit_sgd(X_data.iloc[0:1000], y[0:1000])

print('###### Fit done ######')
preds, prob_here = model.predict(X_data.iloc[800:1371])
print('Final Testing Accuracy')
print(accuracy(preds, y[800:1371], 1371-800))
print('\nAverage Training Accuracy')
print(np.mean(ploty))

tp = truepositives(preds, y[800:1371], 1371-800)
fp = falsepositives(preds, y[800:1371], 1371-800)
fn = falsenegatives(preds, y[800:1371], 1371-800)

precision = tp/(tp+fp)
recall = tp/(tp+fn)
fscore = 2*precision*recall/(precision+recall)

print('Precision: '+str(precision))
print('Recall: '+str(recall))
print('Fscore: '+str(fscore))

loss = model._binary_cross_entropy(y[800:1371], prob_here)
print('Loss is: '+str(loss))

plt.plot(range(0,5001,50), ploty, color = 'blue',
         linestyle = 'solid')
plt.title('SGD 0.001 LR')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
# plt.savefig("sgdpoint001", facecolor='white',
#             pad_inches=0.3, transparent=True)