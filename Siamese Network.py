# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 19:08:19 2018

'''
Group
Vidhi Patel n9807641
Nishit Ramoliya n9915567
IFN680 Assignment 2 
Siamese Network
"""

import numpy as np 
import itertools
from tensorflow import keras
from keras import backend as K
RMSProp = keras.optimizers.RMSprop 
import matplotlib.pyplot as plt

""" 
# split_1 is used for training and testing
# split_2 is used for only testing
"""
split_1 = [2,3,4,5,6,7]
split_2 = [1,0,8,9]
limit = 3000       #code tested for limit 1000 and 2000
totalpair = 3000    #code tested for totalpair 1000 and 2000
batch_size = 128
epochs = 10         #code tested for epoch 20 and 25

    
"""
Task 1: Load MNIST dataset and split data such that 
the digits in [2,3,4,5,6,7] are used for training and testing
the digits in [0,1,8,9] are only used for testing, none of these digits should be used
during training.
"""
def load_data():       
    """
    Load MNIST data from keras.datasets.mnist.load_data
    """
    (x_train,y_train ),(x_test,y_test) = keras.datasets.mnist.load_data()
   
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    """
    Split data as per the assignment requirement
    Split1: 2,3,4,5,6,7
    Split2: 0,1,8,9
    """
    split1_trn = np.isin(y_train,split_1)
    split1_test = np.isin(y_test,split_1)
    split2_test = np.isin(y_test,split_2)
    """
    Train and test set from the split 1
    """
    x_train_split1 = x_train[split1_trn]
    y_train_split1 = y_train[split1_trn]
    x_test_split1 = x_test[split1_test]
    y_test_split1 = y_test[split1_test]
    """
    Test set from the split 2
    """
    x_test_split2 = x_test[split2_test]
    y_test_split2 = y_test[split2_test]    
    
    x_split1 = np.append(x_train_split1,x_test_split1,axis = 0)
    y_split1 = np.append(y_train_split1,y_test_split1,axis = 0)    
    x_split2 = np.append(x_train,x_test_split2,axis = 0) 
    y_split2 = np.append(y_train,y_test_split2,axis = 0)  
    x_trn_final = np.expand_dims(x_split1,-1)
    x_test_final = np.expand_dims(x_split2,-1)    
    y_trn_final = keras.utils.to_categorical(y_split1)
    y_test_final = keras.utils.to_categorical(y_split2)
    
#    print('xtrain, ytrain',x_trn_final,y_trn_final)
#    print('xtrain, ytrain',x_trn_final,y_trn_final)
      
    return  x_trn_final,y_trn_final,x_test_final,y_test_final

"""
task 2: Implement and test contrasive loss function
"""
def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = (0.5) * K.square(y_pred)
    margin_square = (0.5) * K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

"""
Task 3: Build a Siamese network
"""
def siamese_network(input_shape):
         
    image1 = keras.layers.Input(input_shape)
    image2 = keras.layers.Input(input_shape)     
    model = keras.models.Sequential()       
    model.add(keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = input_shape))
    model.add(keras.layers.Conv2D(32,(3,3), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation = 'relu'))   
#    model.compile(loss=contrastive_loss, optimizer='rms', metrics=[accuracy])
   
    encoded_1 = model(image1)
    encoded_2 = model(image2)   
    lambda_result = keras.layers.Lambda(lambda x: K.sqrt(K.maximum(K.sum(K.square(x[0] - x[1]), axis = 1, keepdims=True), K.epsilon())))([encoded_1, encoded_2])
    return keras.Model(inputs=[image1,image2],outputs=lambda_result)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def make_pair(x, y):
    pair = [] 
    label = [] 
    nested_combination = [] 
    indices = np.arange(len(x))
    np.random.shuffle(indices) 
    divide = len(indices)//2
    first_half = indices[divide:]
    second_half = indices[:divide] 
    first_half = first_half[limit:]
    second_half = second_half[:limit]
    merge = [first_half,second_half]
    final_pair = list(itertools.product(*merge))
    np.random.shuffle(final_pair)
    positive_pair = totalpair // 2 -1
    negative_pair = totalpair // 2 -1 
    
    count = 0     
    for img_list_1, img_list_2 in final_pair:
        if positive_pair < count:
            break
        
        if(np.array_equal(y[img_list_1], y[img_list_2])):
            pair += [[x[img_list_1], x[img_list_2]]]
            label += [1]
            nested_combination.append((img_list_1, img_list_2))
            count += 1
  
    count = 0
    for img_list_1, img_list_2 in final_pair:
        if negative_pair < count:
                break
        if(np.array_equal(y[img_list_1], y[img_list_2])):
            continue
        else:
            pair += [[x[img_list_1], x[img_list_2]]]
            label += [0]
            count += 1
            nested_combination.append((img_list_1, img_list_2))
    
    return(np.array(pair),np.array(label),nested_combination)


"""
Task 4: Train a siamese network with data
"""

x_tr,y_tr,x_te,y_te = load_data()
x_train_split1, x_test_split1 = np.split(x_tr, [int(len(x_tr) * 0.8)])
y_train_split1, y_test_split1 = np.split(y_tr, [int(len(y_tr) * 0.8)])
input_shape= x_train_split1.shape[1:]

x_tr_split1_pair, y_tr_split1_pair, split1 = make_pair(x_train_split1, y_train_split1)
x_te_split1_pair, y_te_split1_pair, testing1 = make_pair(x_test_split1, y_test_split1)    
x_te_split1_2_pair, y_te_split1_2_pair, testing2 = make_pair(x_test_split1,y_te)
x_te_split2_pair, y_te_split2_pair, testing3 = make_pair(x_te, y_te)    

train_siamese = siamese_network(input_shape)    
train_siamese.compile(optimizer = RMSProp(), loss = contrastive_loss, metrics=[accuracy])
history=train_siamese.fit([x_tr_split1_pair[:,0],x_tr_split1_pair[:,1]],
                y_tr_split1_pair,
                batch_size= batch_size,
                epochs= epochs,
                validation_data=([x_te_split1_pair[:,0],x_te_split1_pair[:,1]],y_te_split1_pair))
 
"""
Task 5: Evaluate the generalisation capability of the Siamese network
by testing it with warious training and testing pairs.
Plot the results of training and valifation errors
"""

y_pred = train_siamese.predict([x_tr_split1_pair[:,0],x_tr_split1_pair[:,1]])
tr_acc = compute_accuracy(y_tr_split1_pair, y_pred)   
y_pred = train_siamese.predict([x_te_split2_pair[:, 0], x_te_split2_pair[:, 1]])
te_acc = compute_accuracy(y_te_split2_pair, y_pred)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();
    

""" Testing Siamese network with pairs from the set of digits [2,3,4,5,6,7]
"""
print("testing siamese network with split 1: [2,3,4,5,6,7]")    
y_pred = train_siamese.predict([x_te_split1_pair[:,0],x_te_split1_pair[:,1]])
tr_acc = compute_accuracy(y_te_split1_pair, y_pred)
y_pred = train_siamese.predict([x_te_split2_pair[:, 0], x_te_split2_pair[:, 1]])
te_acc = compute_accuracy(y_te_split2_pair, y_pred)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


"""Testing network with pairs from the set of digits [2,3,4,5,6,7] union [0,1,8,9] 
"""   
print("testing siamese network with [2,3,4,5,6,7] union [0,1,8,9]")
y_pred = train_siamese.predict([x_te_split1_2_pair[:,0],x_te_split1_2_pair[:,1]])
tr_acc = compute_accuracy(y_te_split1_2_pair, y_pred)
y_pred = train_siamese.predict([x_te_split2_pair[:, 0], x_te_split2_pair[:, 1]])
te_acc = compute_accuracy(y_te_split2_pair, y_pred)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


"""testing it with pairs from the set of digits [0,1,8,9]
"""   
print("testing siamese network with split 2: [0,1,8,9]")    
y_pred = train_siamese.predict([x_te_split2_pair[:,0],x_te_split2_pair[:,1]])
tr_acc = compute_accuracy(y_te_split2_pair, y_pred)  
y_pred = train_siamese.predict([x_te_split2_pair[:, 0], x_te_split2_pair[:, 1]])
te_acc = compute_accuracy(y_te_split2_pair, y_pred)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();
    



