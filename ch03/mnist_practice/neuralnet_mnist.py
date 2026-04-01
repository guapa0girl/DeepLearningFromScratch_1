# 손글씨 숫자 인식
import sys,os
sys.path.append(os.pardir)

import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image

# 이미지 출력 함수
def img_show(img):
    pil_img=Image.fromarray(np.unit8(img))
    pil_img.show()
    
# 데이터 로드
def get_data():
    (x_train, t_train),(x_test, t_test) = \
        load_mnist(normalize=True, flatten=True)
    return x_test, t_test

# 네트워크 초기화 (학습된 가중치 로드)
def init_network():
    file_path=os.path.join(os.path.dirname(__file__), "dataset","sample_weight.pkl")
    
    with open(file_path,'rb') as f:
        network=pickle.load(f)
    return network

# 활성화 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
# 예측 함수 (순전파)
def predict(network, x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=sigmoid(a3)
    
    return y

# 정확도 평가
x,t= get_data()
network=init_network()

accuracy_cnt=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accuracy_cnt+=1
print("Accuracy: " + str(float(accuracy_cnt)/len(x)))

# 가중치 형상 출력
x,_=get_data()
network=init_network()
W1,W2,W3=network['W1'],network['W2'],network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)

# 배치 처리
x,t = get_data()
network=init_network()

batch_size = 100 # 배치 크기 100 (100으로 묶어서 입/출력 진행)
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt +=np.sum(p==t[i:i+batch_size])
print("Accuracy: " + str(float(accuracy_cnt)/len(x)))