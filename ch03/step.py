def step_function(x):
    if x>0:
        return 1
    else:
        return 0
    
print(step_function(2.3)) #1
#print(step_function(np.array([1.0,2.0]))) # 불가능

def step_func(x):
    y=x>0
    return y.astype(int)

import numpy as np
x=np.array([-1.0,1.0,2.0])
y=x>0
print(x)
print(y)