import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

np.random.seed(101) 
tf.random.set_seed(101)

tf.compat.v1.disable_eager_execution()


x_data = np.linspace(0, 50, 50) 
y_data = np.linspace(0, 50, 50) 
   
x_data += np.random.uniform(-4, 4, 50) 
y_data += np.random.uniform(-4, 4, 50) 
  
length_x = len(x_data)

New_X_data = tf.compat.v1.placeholder("float") 
New_Y_data = tf.compat.v1.placeholder("float") 

var_W = tf.Variable(np.random.randn(), name = "W") 
var_b = tf.Variable(np.random.randn(), name = "b") 

rate_of_learning = 0.01
epochs_for_training = 500

y_predictions = tf.add(tf.multiply(New_X_data, var_W), var_b) 
  
function_cost = tf.reduce_sum(tf.pow(y_predictions-New_X_data, 2)) / (2 * length_x) 
  
optimizer_gradient = tf.compat.v1.train.GradientDescentOptimizer(rate_of_learning).minimize(function_cost) 
#optimizer_gradient = tf.compat.v1.train.ProximalGradientDescentOptimizer(rate_of_learning).minimize(function_cost)
  
variable_initialize = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess: 
      
    sess.run(variable_initialize) 
      
    for epoch in range(epochs_for_training):
      for (x_loop, y_loop) in zip(x_data, y_data):
        sess.run(optimizer_gradient, feed_dict = {New_X_data: x_loop, New_Y_data: y_loop})
        if (epoch + 1) % 50 == 0:
          c = sess.run(function_cost, feed_dict = {New_X_data : x_loop, New_Y_data : y_loop}) 
          print("Epoch", (epoch + 1), ": cost =", c, "var_1 =", sess.run(var_W), "var_2 =", sess.run(var_b)) 
      
    cost_training = sess.run(function_cost, feed_dict ={New_X_data: x_loop, New_Y_data: y_loop}) 
    weight = sess.run(var_W) 
    bias = sess.run(var_b)

make_predictions = weight * x_data + bias 
print("Training cost =", cost_training, "Weight =", weight, "bias =", bias, '\n')
     
plt.plot(x_data, y_data, 'bo',label ='Sample data taken') 
plt.plot(x_data, make_predictions, label ='line fitted') 
plt.title('Result for linear regression') 
plt.legend() 
plt.show()
     