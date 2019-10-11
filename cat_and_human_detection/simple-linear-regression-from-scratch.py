## import the modules
import numpy as np
import tensorflow as tf

## create the place holder

X = tf.placeholder('float') ## create the bucket where you put the value
Y = tf.placeholder('float')

X_train = np.asarray([1,2.2,3.3,4.1,5.2])  ## x_train data 6 value
Y_train = np.asarray([2,3,3.3,4.1,3.9,1.6]) ## Ytrain 1 more thats we gonna predict

## make the model the model with be y=mx+b

def model(X,w):
    return tf.multiply(X,w)


## put the weight of the graph

w = tf.Variable(0.0,name='weights') ## default is 0.0 we gonna update it later

y_model = model(X,w)   ## this is the model

## find the cost function to evalute the error
# the cost function will be mse
cost = (tf.pow(Y-y_model,2))


## now we have to train the model with Gradient Desent Optimizer
lr=0.01
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)


sess = tf.Session();
init = tf.initialize_all_variables()
sess.run(init)



## ok the graph is done now run through a loop

for trails in range(50): ## this is the epochs
    
    for(x,y) in zip(X_train,Y_train): ## take the x and the y of the data
        sess.run(train_op,feed_dict={X:x,Y:y})

## at last print the final weight
print(sess.run(w)) 