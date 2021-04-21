# libs
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#np.random.seed(42)

### PRE PROCESSING ###
df = pd.read_csv('dataset_NN.csv',header=None)
df.columns = ['x1','x2','x3','x4','x5','x6','label']
df = df[1:]
print(df.head())
numlabels = 10

def normalize(ds):
    for col in ds.columns:
        ds[col] = ds[col].astype(float)
    for itr in range(ds.shape[1]):
        if ds.columns[itr] == 'label' or ds.columns[itr] == 'x1' or ds.columns[itr] == 'x2'or ds.columns[itr] == 'x3' or ds.columns[itr] == 'x0' :
        	continue
        original = ds.iloc[:,itr]
        ds.iloc[:,itr] = (original - np.mean(original))/np.std(original)
    return ds

df = normalize(df)
print(df.head())
df,df_test = np.split(df.sample(frac=1).reset_index(drop=True),[int(.7*len(df))])

### LABELS ###
labels = []
for row in range(0,df.shape[0]):
	labels.append(df.iloc[row,6])

one_hot_labels = np.zeros((df.shape[0], numlabels))
for i in range(0,df.shape[0]):
    one_hot_labels[i, int(labels[i])-1] = 1

test_labels = []
for row in range(0,df_test.shape[0]):
	test_labels.append(df_test.iloc[row,6])

one_hot_two = np.zeros((df_test.shape[0], numlabels))
for i in range(0,df_test.shape[0]):
    one_hot_two[i, int(test_labels[i])-1] = 1

feature_set = df.drop('label',axis=1)
feature_set = feature_set.values
feature_set = feature_set.astype(np.float)
print(type(feature_set))
feature_set_test = df_test.drop('label',axis=1)
feature_set_test = feature_set_test.values
feature_set_test = feature_set_test.astype(np.float)

### START ANN ###
def sigmoid(x):
	z = np.exp(-x)
	sig = 1 / (1 + z)
	return sig

def derivative_sigmoid(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    exps = np.exp(A - np.max(A, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def softmax_predict(A):
	exps = np.exp(A-np.max(A))
	return exps/np.sum(exps)

def feedforward(feature_set,wo,bo,wh,bh):
    zh = np.dot(feature_set, wh) + bh
    ah = sigmoid(zh)
    # Phase 2
    before_activation_outputs = np.dot(ah, wo) + bo
    predicted_labels = softmax_predict(before_activation_outputs)
    return predicted_labels.argmax()

def feedforward_2(feature_set,wo,bo,w2,b2,w1,b1):
    zh = np.dot(feature_set, w1) + b1
    ah = sigmoid(zh)
    # Phase 2
    zh2 = np.dot(ah,w2) + b2
    ah2 = sigmoid(zh2)
    # Phase 3
    before_activation_outputs = np.dot(ah2, wo) + bo
    predicted_labels = softmax_predict(before_activation_outputs)
    return predicted_labels.argmax()

def predict(feature_set,wo,bo,w2=[],b2=[],w1=[],b1=[]):
	if len(w1)>=1:
		outs = feedforward_2(feature_set,wo,bo,w2,b2,w1,b1)
		return outs
	outs = feedforward(feature_set,wo,bo,w2,b2)
	return outs

def get_acc(x, y,wo,bo,wh=[],bh=[],w1=[],b1=[]):
    acc = 0
    for xx,yy in zip(x, y):
        s = predict(xx,wo,bo,wh,bh,w1,b1)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

def buildModel(feature_set,num_hidden_nodes,layers,lr,epochs):
	num_samples = feature_set.shape[0]
	num_features = feature_set.shape[1]
	num_classes = 10
	feature_set_original = feature_set
	global one_hot_labels
	one_hot_labels_original = one_hot_labels
	# Output layer
	if layers==2:
		wo = np.random.rand(num_hidden_nodes,num_classes)
		bo = np.random.randn(num_classes)	
		w2 = np.random.rand(num_hidden_nodes,num_hidden_nodes)
		b2 = np.random.randn(num_hidden_nodes)
		w1 = np.random.rand(num_features,num_hidden_nodes)
		b1 = np.random.randn(num_hidden_nodes)
		error_cost = []
		acc_history = []
		for epoch in range(epochs):
		    batch_size = 50  # for 2 random indices
		    index = np.random.choice(feature_set.shape[0], batch_size, replace=False) 
		    feature_set = feature_set_original[index]
		    one_hot_labels = one_hot_labels_original[index]
		    #feedforward
		    zh = np.dot(feature_set, w1) + b1
		    ah = sigmoid(zh)
		    
		    zh2 = np.dot(ah,w2) + b2
		    ah2 = sigmoid(zh2)
		    
		    before_activation_outputs = np.dot(ah2, wo) + bo
		    predicted_labels = softmax(before_activation_outputs)
		    #Back Propagation
		    a3_delta= predicted_labels - one_hot_labels # cross entropy

		    z2_delta = np.dot(a3_delta,wo.T)
		    a2_delta = z2_delta * derivative_sigmoid(ah2)
		    z1_delta = np.dot(a2_delta,w2.T)
		    a1_delta = z1_delta * derivative_sigmoid(ah)
		    #Grad desc
		    wo -= lr*np.dot(ah2.T,a3_delta)
		    bo -= lr*np.sum(a3_delta,axis=0)
		    w2 -= lr * np.dot(ah.T,a2_delta)
		    b2 -= lr * np.sum(a2_delta,axis=0)
		    w1 -= lr * np.dot(feature_set.T,a1_delta)
		    b1 -= lr * np.sum(a1_delta,axis = 0)

		    if epoch % 50 == 0:
		        loss = np.sum(-one_hot_labels * np.log(predicted_labels))/feature_set.shape[0]
		        print('Loss function value: ', loss)
		        error_cost.append(loss)
		        acc = get_acc(feature_set,one_hot_labels,wo,bo,w2,b2,w1,b1)
		        print('Accuracy value: ', acc)
		        acc_history.append(acc)

		return acc_history,error_cost,wo,bo,w2,b2,w1,b1
	else:
		wh = np.random.rand(num_features,num_hidden_nodes)
		bh = np.random.randn(num_hidden_nodes)
		wo = np.random.rand(num_hidden_nodes,num_classes)
		bo = np.random.randn(num_classes)
		error_cost = []
		acc_history = []
		for epoch in range(epochs):
		    batch_size = 50  # for 2 random indices
		    index = np.random.choice(feature_set_original.shape[0], batch_size, replace=False) 
		    feature_set = feature_set_original[index]
		    one_hot_labels = one_hot_labels_original[index]
		    #feedforward 
		    zh = np.dot(feature_set, wh) + bh
		    ah = sigmoid(zh)
		    
		    before_activation_outputs = np.dot(ah, wo) + bo
		    predicted_labels = softmax(before_activation_outputs)
		    # BACK PROP
		    delta1 = predicted_labels - one_hot_labels
		    output_layer_updates = np.dot(ah.T, delta1)
		    output_bias_upadtes = delta1
		
		    del_wT = np.dot(delta1 , wo.T)
		    sig_del_wT = derivative_sigmoid(zh)
		    hidden_weight_updates = np.dot(feature_set.T, sig_del_wT * del_wT)
		    hidden_bias_updates = del_wT * sig_del_wT
		    # Grad Desc
		    wh -= lr * hidden_weight_updates
		    bh -= lr * hidden_bias_updates.sum(axis=0)
		    wo -= lr * output_layer_updates
		    bo -= lr * output_bias_upadtes.sum(axis=0)

		    if epoch % 50 == 0:
		        loss = np.sum(-one_hot_labels * np.log(predicted_labels))/feature_set.shape[0]
		        print('Loss function value: ', loss)
		        error_cost.append(loss)
		        acc = get_acc(feature_set,one_hot_labels,wo,bo,wh,bh)
		        print('Accuracy value: ', acc)
		        acc_history.append(acc)

		return acc_history,error_cost,wo,bo,wh,bh

feature_set_original = feature_set
one_hot_labels_original = one_hot_labels
#acc_history,error_cost,wo,bo,wh,bh = buildModel(feature_set,32,1,0.01,5000)

acc_history,error_cost,wo,bo,w2,b2,w1,b1 = buildModel(feature_set_original,128,2,0.01,50000)
print(get_acc(feature_set_test,one_hot_two,wo,bo,w2,b2,w1,b1))
### RESULTS ###
print("Training accuracy : ")
print(get_acc(feature_set_original,one_hot_labels_original,wo,bo,wh,bh))
print("Testing accuracy : ")
print(get_acc(feature_set_test,one_hot_two,wo,bo,wh,bh))

### PLOTS ####
epoch_count = []
for i in range(1,len(acc_history)+1):
	epoch_count.append(50*i)

plt.plot(epoch_count,error_cost)
plt.xlabel("No of Iterations")
plt.ylabel("Loss")
plt.title("Loss plot")
plt.show()
#plt.savefig('One Layer Loss plot.png', bbox_inches='tight')
plt.close()

plt.plot(epoch_count,acc_history)
plt.xlabel("No of Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy plot")
plt.show()
#plt.savefig('One Layer Accuracy plot.png', bbox_inches='tight')
plt.close()

lrs = [0.01,0.001,0.0001]

for lr in lrs:
	one_hot_labels = one_hot_labels_original
	acc_history,error_cost,wo,bo,wh,bh = buildModel(feature_set_original,128,1,lr,5000)
	print("Accuracy PLOT")
	plt.plot(epoch_count,acc_history)
	plt.xlabel("No of Iterations")
	plt.ylabel("Accuracy")
	plt.title("Accuracy plot for learning rate = "+str(lr)+" and layers = 1")
	#plt.savefig('One Layer and Learning rate '+str(lr)+' '+'Hidden Nodes 32'+'.png', bbox_inches='tight')
	plt.close()

for lr in lrs:
	one_hot_labels = one_hot_labels_original
	acc_history,error_cost,wo,bo,w2,b2,w1,b1 = buildModel(feature_set_original,256,2,lr,5000)
	print("Accuracy PLOT")
	plt.plot(epoch_count,acc_history)
	plt.xlabel("No of Iterations")
	plt.ylabel("Accuracy")
	plt.title("Accuracy plot for learning rate = "+str(lr)+" and layers = 2")
	#plt.savefig('Two Layers and Learning rate '+str(lr)+' '+'Hidden Nodes 64'+'.png', bbox_inches='tight')
	plt.close()