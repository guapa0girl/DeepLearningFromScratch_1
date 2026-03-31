#3.4.2. 뉴런 각 층의 신호 전달 구현하기
# input -> hidden1 신호 전달
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])

#print(W1.shape)
#print(X.shape)
#print(B1.shape)

A1=np.dot(X,W1)+B1
Z1=sigmoid(A1)
print(Z1)


#은닉층 hidden layer1->2 신호 전달

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

#print(Z1.shape)
#print(W2.shape)
#print(B2.shape)

A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)
print(Z2)

#hidden 2 -> out 신호 전달
def identify_function(x):
    return x
W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y=identify_function(A3) # Y = A3