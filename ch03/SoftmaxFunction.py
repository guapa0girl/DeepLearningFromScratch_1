import numpy as np
import matplotlib.pylab as plt

a=np.array([0.3,2.9,4.0])

#exp_a=np.exp(a)
#print(exp_a)

#sum_exp_a=np.sum(exp_a)
#print(sum_exp_a)

#y=exp_a/sum_exp_a
#print(y)

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) # overflow 대책, 결과는 같다.
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    
    return y
y=softmax(a)
print(y)
print(np.sum(y))

#overflow 예시
#b=np.array([1010,1000,990])
#print(np.exp(b)/np.sum(np.exp(b)))

#c=np.max(b)
#print(b-c)
#print(np.exp(b-c)/np.sum(np.exp(b-c)))