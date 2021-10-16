import numpy as np
import pandas as pd
import time

# N_ROW = 4
# N_COL = 12
# N_SPA = N_ROW*N_COL
# ACTIONS = ['l','r','u','d']
# E = 0.9
# ALPHA = 0.1
# GAMMA = 1
# MAX_TRAN = 36

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
print(df.loc[1,'b'])
# df = df.sample(n=1, axis=1)
# print(df)
# a='test'
# b=2
# print('Eposode {},{}'.format(a,b))