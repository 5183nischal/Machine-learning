import numpy as np
import math

y = [1,0,0,0,1,0,1,1,1,0]
for i in range(len(y)):
	y[i] = 2*y[i] -1
print(y)