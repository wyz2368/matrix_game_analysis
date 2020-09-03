from numpy import genfromtxt
import numpy as np

my_data = genfromtxt('Book1.csv', delimiter=',')
print(np.shape(my_data))