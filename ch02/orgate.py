import numpy as np
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.1
    tmp=np.sum(x*w)+b
    if tmp>0:
        return 1
    else:
        return 0
print(OR(1,1)) #1
print(OR(0,1))
print(OR(1,0))
print(OR(0,0)) #0