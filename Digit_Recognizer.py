#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read csv using pandas
df=pd.read_csv("/Users/thushara/Documents/Thushara/SMIT/PROJECTS/Digit_Recognizer/train.csv")
print(df.head())

#convert df into an array
df=np.array(df)
#obtaining number of rows as m, number of columns as n
m,n= df.shape
#shuffling the dataset into a random order
np.random.shuffle(df)
#splitting the dataset into train and test data
df_test=df[0:1000].T
df_train=df[1000:m].T
#obtaining features and results column to train and test model
y_test=df_test[0]
x_test=df_test[1:n]
y_train=df_train[0]
x_train=df_train[1:n]
x_train=x_train/255
x_test=x_test/255

def init_parameters():
    w1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    w2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return w1,b1,w2,b2

def ReLU(z):
    return np.maximum(0,z)
    # if z>0:
    #     return z
    # else:
    #     return 0

def der_ReLU(z):
    return(z>0)

def softmax(z):
    return (np.exp(z)/(sum(np.exp(z))))

def one_hot_encoding(y):
    one_hot_y=np.zeros((y.size,y.max()+1))
    one_hot_y[np.arange(y.size),y]=1
    one_hot_y=one_hot_y.T
    return one_hot_y

def forward_propogation(w1,b1,w2,b2,x):
    Z1=np.dot(w1,x)+b1
    A1=ReLU(Z1)
    Z2=np.dot(w2,A1)+b2
    A2=softmax(Z2)
    return Z1,A1,Z2,A2


def back_propogation(Z1,A1,Z2,A2,w1,w2,x,y):
    Y=one_hot_encoding(y)
    dZ2=A2-Y
    dw2=(1/m)*(np.dot(dZ2,A1.T))
    db2=(1/m)*(np.sum(dZ2))
    dZ1=np.dot(w2.T,dZ2)*der_ReLU(Z1)
    dw1=(1/m)*(np.dot(dZ1,x.T))
    db1=(1/m)*(np.sum(dZ1))
    return dw1,db1,dw2,db2
    
def update_parameters(w1,b1,w2,b2,dw1,db1,dw2,db2,lr):
    w1=w1-dw1*lr
    b1=b1-db1*lr
    w2=w2-dw2*lr
    b2=b2-db2*lr
    return w1,b1,w2,b2

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return(np.sum(predictions==Y)/Y.size)

def get_predictions(A2):
    return np.argmax(A2,0)

def gradient_descent(X,Y,lr,n_iter):
    w1,b1,w2,b2=init_parameters()
    for i  in range(n_iter):
        Z1,A1,Z2,A2=forward_propogation(w1,b1,w2,b2,X)
        dw1,db1,dw2,db2=back_propogation(Z1,A1,Z2,A2,w1,w2,X,Y)
        w1,b1,w2,b2=update_parameters(w1,b1,w2,b2,dw1,db1,dw2,db2,lr)
        if i % 10==0:
            print("Iteration= ",i)
            print("Accuracy= ",get_accuracy(get_predictions(A2),Y) )
    return w1,b1,w2,b2


w1, b1, w2, b2=gradient_descent(x_train,y_train,1,500)


def make_predictions(X, w1, b1, w2, b2):
    _, _, _, A2 = forward_propogation(w1, b1, w2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, w1, b1, w2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], w1, b1, w2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()