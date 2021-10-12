import pandas as pd
data = pd.read_csv("data.csv",header= None)
#data.columns=["char_a","char_b","char_c","char_d","Result"]
data_lst=[]
for i in range(9):
    temp_lst=[1]
    for j in range(2):
        temp_lst.append(data.iloc[i,j])

    data_lst.append(temp_lst)

desired = []
for i in range(9):
    if data.iloc[i,2] == 0:
        desired.append([1,0,0])
    if data.iloc[i,2] == 1:
        desired.append([0,1,0])
    if data.iloc[i,2] == 2:
        desired.append([0,0,1])

#print(data_lst[149])
