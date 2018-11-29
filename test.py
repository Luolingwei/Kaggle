import pandas as pd
import numpy as np

dic={0:'A',1:'B',2:'C',3:'D'}
frame=pd.DataFrame(np.random.randn(4,4),columns=['A','B','C','D'])
series=pd.Series(np.arange(4))
T=series.map(dic)
print(T)
