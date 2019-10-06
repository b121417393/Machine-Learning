import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import TEST_Dataset
CUDA_DEVICES = 0
DATASET_ROOT = './modified_test'
PATH_TO_WEIGHTS = './mymodel.pth'


def test():

    data_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		
    test_set = TEST_Dataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=1)

    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    testing_loss = 0.0
    total_correct = 0
    total = 0
	
    criterion = nn.CrossEntropyLoss()
	
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
			
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            testing_loss += loss.item() * inputs.size(0)
			

            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(labels.size(0)):
                label =labels[i]

    testing_loss = testing_loss / len(test_set)
    
    print("")
    print("Total Correct: ", total_correct)
    print("Accuracy on the ALL test images: ", '%2.4f' %(100 * total_correct / total), '%')
    print("Testing loss: ", '%2.6f' %testing_loss)
    print("")



if __name__ == '__main__':
    test()
