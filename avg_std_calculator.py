import numpy as np


df_0= np.loadtxt("0.txt")
df_1= np.loadtxt("1.txt")
df_2=np.loadtxt("2.txt")
df_3 = np.loadtxt("3.txt")
df_4=np.loadtxt("4.txt")

rows=[]
rows=np.array([df_0[0]])
rows=np.concatenate(rows, np.array([df_1[0]]), axis=0)
rows=np.concatenate(rows, np.array([df_2[0]]), axis=0)
rows=np.concatenate(rows, np.array([df_3[0]]), axis=0)
rows=np.concatenate(rows, np.array([df_4[0]]), axis=0)
print(rows)
average=np.average(rows, axis=0)
std=np.average(rows, axis=0)
print(average)
print(std)
np.sa