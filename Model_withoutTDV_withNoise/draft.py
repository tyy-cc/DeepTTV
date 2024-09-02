import numpy as np
import matplotlib.pyplot as plt

filename1 = 'lr_curve.txt'
filename2 = 'lr_curve_1.txt'

data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)

epoch1 = data1[:, 0]
train_error1= data1[:, 1]

epoch2 = data2[:, 0]
train_error2= data2[:, 1]

plt.plot(epoch1, train_error1, alpha=0.5)
plt.plot(epoch2, train_error2, alpha=0.5)
plt.yscale('log')
plt.ylabel('train loss')
plt.xlabel('epoch')
plt.savefig('compare_train_loss.png')


