import numpy as np
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7 # 편향
    tmp = np.sum(x*w)+b
    print(tmp)
    if tmp>0 :
        return 1
    else:
        return 0
print(AND(0,1)) # 0
print(AND(1,1)) # 1