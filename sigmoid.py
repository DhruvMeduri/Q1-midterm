import q5
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import matplotlib.pyplot as plt
#this code has 4 neurons in the hidden layer(excluding bias), 2 input neurons(excluding bias) and 3 output neurons
#3X4 matrix(including the bias) for input to hidden connection and 5X3 for hidden to output multilayer

#a = np.zeros([3,4])
#b = np.zeros([5,3])
a = np.random.rand(3,4)
b = np.random.rand(5,3)
a_list = [a,a]# this stores the list of a matrices needed to compute the momentum term in delta
b_list = [b,b]# this stores the list of b matrices needed to compute the momentum term in delta
err = []
count = 0
full_order = np.random.permutation(9)
'''
training = []
testing = []
err = []
for f in range(100):# using 100 random data points for training
    training.append(full_order[f])
for g in range(100,150):# using 50 random data points for testing
    testing.append(full_order[g])
'''
for epoch in range(5000):# we run 10000 epochs
    for i in full_order:
        count = count + 1
        temp_inp = np.array(q5.data_lst[i])
        #this is for the forward propagation
        hidden_output = np.dot(temp_inp,a)#1X4 vector
        hidden_activation = np.array([1,0,0,0,0],dtype = 'float')
        for j in range(1,5):
            hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X5 vector(first element is the bias), these serve as inputs for the output layer
        output = np.dot(hidden_activation,b)#this is a 1X3 matrix
        final_output = np.array([0,0,0],dtype = 'float')
        for j in range(3):
            final_output[j] = 1/(1 + np.exp(-output[j])) #1X3 vector, this is the final_output

        #this completes the forward propagation, now we implement the backward propagation
        des = np.array(q5.desired[i],dtype = 'float')
        #first we compute the local gradient of the output nodes
        output_loc_grad = np.multiply(np.multiply((des - final_output),final_output),(np.array([1,1,1])-final_output))#1X3 vector
        #now to calculate the local gradient of the neurons in the hidden layer
        bT = np.transpose(b)
        #temp_act = np.array([hidden_activation[1],hidden_activation[2]])#excludes the bias from the neurons in the hidden layer
        hidden_loc_grad = np.multiply(np.multiply(np.dot(output_loc_grad,bT),hidden_activation),(np.array([1,1,1,1,1]) - hidden_activation))#1X4 vector
        hidden_loc_grad1 = np.array([hidden_loc_grad[1],hidden_loc_grad[2],hidden_loc_grad[3],hidden_loc_grad[4]])#this is a 1X4 vector which removes the bias neuron

        # now to update the weights after every Iterations
        learn = 2.5
        alpha = 0.1
        #first we update the weights matrix 'a'
        for r in range(3):
            for c in range(4):
                a[r][c] = a[r][c] + ((a_list[count][r][c] - a_list[count-1][r][c])*alpha) + learn*hidden_loc_grad1[c]*temp_inp[r]# includes the momentum term

        # now we update the weights matrix 'b'
        for r in range(5):
            for c in range(3):
                b[r][c] = b[r][c] + ((b_list[count][r][c] - b_list[count-1][r][c])*alpha) + learn*output_loc_grad[c]*hidden_activation[r]#includes the momentum term
        a_list.append(a)
        b_list.append(b)
        #print("matrix A: ",a)
        #print("Matrix B: ",b)
#this is for plotting the error trajectory
    if epoch%500 == 0:
        error = 0
        for t in full_order:
            temp_inp = np.array(q5.data_lst[t])
            des = np.array(q5.desired[t])
            hidden_output = np.dot(temp_inp,a)#1X8 vector
            #print(temp_inp)
            #print(hidden_output)
            #print(hidden_output)
            hidden_activation = np.array([1,0,0,0,0],dtype = 'float')
            for j in range(1,5):
                hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X9 vector(first element is the bias), these serve as inputs for the output layer
            #print(hidden_activation)
            output = np.dot(hidden_activation,b)#this is a 1X3 matrix
            final_output = np.array([0,0,0],dtype = 'float')
            #print(final_output)
            for j in range(3):
                final_output[j] = 1/(1 + np.exp(-output[j])) #1X3 vector, this is the final_output
            if np.argmax(des) != np.argmax(final_output):
                error = error + 1
        err.append(error)

#plot code
x_list = []
for i in range(1, 5001):
    if i%500 == 0:
       x_list.append(i)
plt.scatter(x_list,err)
plt.plot(x_list,err)
plt.xlabel("Epochs")
plt.ylabel("%error")
plt.title("Error Trajectory")
plt.show()

# This is for plotting the decision boundaries
xc1 = [0,1,2]
yc1 = [0,2,1]
xc2 = [0,1,2]
yc2 = [1,0,2]
xc3 = [0,1,2]
yc3 = [2,1,0]
plt.scatter(xc1,yc1,color = 'blue')
plt.scatter(xc2,yc2,color = 'red')
plt.scatter(xc3,yc3,color = 'green')
x = np.linspace(-1,4,100)
y = (-a[1][0]/a[2][0])*x + (-a[0][0]/a[2][0])
plt.plot(x,y,color='blue')
y = (-a[1][1]/a[2][1])*x + (-a[0][1]/a[2][1])
plt.plot(x,y,color = 'green')
y = (-a[1][2]/a[2][2])*x + (-a[0][2]/a[2][2])
plt.plot(x,y,color = 'red')
y = (-a[1][3]/a[2][3])*x + (-a[0][3]/a[2][3])
plt.plot(x,y, color = 'black')
plt.title("Decision Boundary")
plt.show()
