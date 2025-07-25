import numpy as np
import pandas as pd 


np.set_printoptions(suppress=True)
csv_file_path = 'C:\\Users\\kerem\\Housing.csv'
df = pd.read_csv(csv_file_path)

data_array = df.to_numpy()

x_train = data_array[:, 1:5]
y_train = data_array[:, 0]
x_train = x_train.astype("float")

w = np.array([0,0,0,0], dtype="float")
b = 0

for i in range(x_train.shape[0]):
    x_train[i][0] = x_train[i][0]/16200.0
    x_train[i][1] = x_train[i][1]/6.0
    x_train[i][2] = x_train[i][2]/4.0
    x_train[i][3] = x_train[i][3]/4.0

for i in range(y_train.shape[0]):
    y_train[i] = y_train[i]/13300000.0

def mean_squared(x,y):
    global w
    global loss_val
    loss_val=0
    for i in range (x.shape[0]):
        loss_val += y[i] - np.dot(x[i],w)
    return loss_val/x.shape[0]

def gradient_descent(x,y,lr):
    global w
    global b
    for j in range(w.shape[0]):
        sum_lossb = 0
        sum_lossw = 0
        for i in range(x.shape[0]):
            sum_lossw+=((np.dot(x[i], w)) - y[i])*x[i][j]
            sum_lossb+=((np.dot(x[i], w)) - y[i])
        w[j] = w[j] - (lr/x.shape[0]) * sum_lossw
        
        print(f"W{j} Cost: {mean_squared(x_train,y_train)}")
    b = b - (lr/x.shape[0]) * sum_lossb

epochs = 3000
learning_rate = 0.003
loss_val = 0


for e in range(epochs):
    gradient_descent(x_train,y_train,learning_rate)
    print(f"Epoch: {e}   , w_1: {w[0]} , w2: {w[1]} , w3: {w[2]} , w4: {w[3]}")
print(f"2. House price predict: {(np.dot(x_train[4], w) + b) * 13300000.0} , real price {y_train[4] * 13300000.0}")
