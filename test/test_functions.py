import numpy as np
from AudioStudio.functions import *

inv_res=100
func=lambda x,s:np.sin(inv_res*x/s)
test_data=normalize_seq([np.sum([func(i,j) for j in range(50,200,50)]) for i in range(5000)])
plt.plot(test_data[0:60]);plt.show()
quick_example(test_data)
