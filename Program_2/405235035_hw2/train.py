import torch
import torch.nn as nn
from dataset import TRAIN_Dataset
from test import test
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import torchvision.models as models
import numpy as np
import copy

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CUDA_DEVICES = 0
DATASET_ROOT = './modified_train'

def train():
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	train_set = TRAIN_Dataset(Path(DATASET_ROOT), data_transform)

	data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)

	model = models.resnet101(pretrained=True)
	fc_features = model.fc.in_features
	model.fc = nn.Linear(fc_features,196)
    
	model = model.cuda(CUDA_DEVICES)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	num_epochs = 50
	criterion = nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0002)

	for epoch in range(num_epochs):
		print("Epoch:", (epoch + 1), '/', (num_epochs))
		print("-----------------------------")

		training_loss = 0.0
		training_corrects = 0

		for i, (inputs, labels) in enumerate(data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()

			outputs = model(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			training_corrects += torch.sum(preds == labels.data)


		training_loss = training_loss / len(train_set)
		training_acc = training_corrects.double() / len(train_set)

		print("Training loss: ", '%2.6f' %training_loss, "   accuracy: ", '%2.6f' %training_acc)

		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())
			
		if epoch == 0:
			torch.save(model, "./mymodel.pth")
			test()
			
		if epoch%10 == 4:
			torch.save(model, "./mymodel.pth")
			test()			
			
		if epoch%10 == 9:
			torch.save(model, "./mymodel.pth")
			test()

	model.load_state_dict(best_model_params)
	torch.save(model, "./best_train_acc.pth")


if __name__ == '__main__':
	train()
