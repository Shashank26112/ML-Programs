import pandas as pd
import numpy as np
data = pd.read_csv("C/MCA/Desktop/MLLab Dataset/enjoysport.csv")
#print(data)
concepts = np.array(data)[:,:-1]
#print(concepts)
target = np.array(data)[:,-1]
#print(target)
def train(concepts,target):
    for i,val in enumerate(target):
        if val=='yes':
            specific_h = concepts[i].copy()
            break
    for i,val in enumerate(concepts):
        if target[i]=='yes':
            for x in range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]='?'
                else:
                    pass

    return specific_h
print(train(concepts,target))