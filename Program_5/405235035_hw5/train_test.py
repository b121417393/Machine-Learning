import torch
import torch.nn as nn
from torch.autograd import Variable
from models import test_LSTM
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from preprocess import data_split,normalize_data
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = './'

df = pd.read_csv("./STT.csv", index_col = 0)

STT = df[df.symbol == 'STT'].copy()
#print(GOOG)
STT.drop(['symbol'],1,inplace=True)
STT_new = normalize_data(STT)
#print(GOOG_new)
window = 30
X_train, y_train, X_test, y_test = data_split(STT_new, window)

INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 1

learning_rate = 0.001
num_epochs = 200

rnn = test_LSTM(input_dim=INPUT_SIZE,hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_dim=OUTPUT_SIZE)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

rnn.to(device)
rnn.train()

train_loss = []
test_loss = []
total_train_error_sum = []
total_test_error_sum = []

for epoch in range(num_epochs):


    error_train_sum = 0
    for inputs, label in zip(X_train,y_train):
        inputs = torch.from_numpy(inputs).float().to(device)
        label = torch.from_numpy(np.array(label)).float().to(device)
        optimizer.zero_grad()

        output =rnn(inputs) # forward
        error_train_sum += (output - label)*(output - label)
        loss=criterion(output,label) # compute loss
        loss.backward() #back propagation
        optimizer.step() #update the parameters
		
    train_loss.append(loss.item())
    print('epoch {}, loss {}'.format(epoch,loss.item()))
    total_train_error_sum.append(error_train_sum)


    error_test_sum = 0
    result = []
    result.clear()
    with torch.no_grad():
        for test_inputs, test_label in zip(X_test,y_test):
            test_inputs = torch.from_numpy(test_inputs).float().to(device)
            test_label = torch.from_numpy(np.array(test_label)).float().to(device)
            test_output =rnn(test_inputs)
            result.append(test_output)
            error_test_sum += (test_output - test_label)*(test_output - test_label)
            loss=criterion(test_output,test_label)
        result =np.array(result)
        test_loss.append(loss.item())
        total_test_error_sum.append(error_test_sum)

    plt.plot(result,color='red', label='Prediction')
    plt.plot(y_test,color='blue', label='Actual')
    plt.legend(loc='best')
    plt.savefig("%d-Prediction.jpg"%epoch)
    plt.close('all')
	

train_loss =np.array(train_loss)
test_loss =np.array(test_loss)
plt.plot(train_loss,color='red', label='Training Loss')
plt.plot(test_loss,color='blue', label='Testing Loss')
plt.legend(loc='best')
plt.savefig("./Loss.jpg")
plt.close('all')


total_train_error_sum =np.array(total_train_error_sum)
total_test_error_sum =np.array(total_test_error_sum)
plt.plot(total_train_error_sum,color='red', label='Train_Total_Error')
plt.plot(total_test_error_sum,color='blue', label='Test_Total_Error')
plt.legend(loc='best')
plt.savefig("Error.jpg")