# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torchsummary import summary
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix
from hp import HPChromosome
import math

def calc_output_size(input_dim, num_conv, num_kernels, kernel_size, conv_stride, num_pooling, pool_size, pool_stride, padding):
    out_dim = input_dim
    while num_conv > 0 or num_pooling > 0:
        if num_conv > 0:
            out_dim = out_dim - kernel_size
            out_dim = out_dim + 2*padding
            out_dim = out_dim // conv_stride
            out_dim = out_dim + 1
            if out_dim < 1:
                return 0
            num_conv = num_conv - 1
        if num_pooling > 0:
            out_dim = out_dim - pool_size
            out_dim = out_dim // pool_stride
            out_dim = out_dim + 1
            if out_dim < 1:
                return 0
            num_pooling = num_pooling - 1
    out_dim = int(out_dim * out_dim * num_kernels)
    return out_dim

def get_activation(num):
    if num == 0:
        return nn.ReLU()
    if num == 1:
        return nn.Tanh()
    if num == 2:
        return nn.ELU()
    if num == 3:
        return nn.Sigmoid()
def get_pool_type(num, kernel_size, stride):
    if num == 0:
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    if num == 1:
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    '''
    if num == 2:
        return nn.FractionalMaxPool2d(kernel_size=kernel_size, stride=stride)
    if num == 3:
        return nn.AdaptiveMaxPool2d(kernel_size=kernel_size, stride=stride)
    '''

def get_optimizer_type(num, learning_rate, momentum, model, l2_pen):
    if num == 0:
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_pen)
    if num == 1:
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_pen, momentum = momentum)
    if num == 2:
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_pen, momentum=momentum)

class LeNet5(nn.Module):
    def __init__(self, num_classes, input_dim, num_conv, num_kernels, kernel_size, conv_stride, num_pooling, pool_size, pool_stride, num_dense, num_neurons, padding, activation_fun, pool_type, dropout, dropout_rate, batch_norm):
        super().__init__()

        con_out = calc_output_size(input_dim, num_conv, num_kernels, kernel_size, conv_stride, num_pooling, pool_size, pool_stride, padding)
        self.valid = True
        if con_out == 0 or num_conv == 0 or num_dense == 0:
            self.valid = False
        if pool_type >1 or pool_type < 0:
            self.valid = False
        if activation_fun >3 or activation_fun < 0:
            self.valid = False


        layers = []
        layers.append(nn.Conv2d(1, num_kernels, kernel_size=kernel_size, stride=conv_stride, padding=padding))
        layers.append(get_activation(activation_fun))
        num_conv = num_conv - 1
        while num_conv > 0 or num_pooling > 0:
            if num_pooling > 0:
                layers.append(get_pool_type(pool_type, pool_size, pool_stride))
                num_pooling = num_pooling - 1
            if num_conv > 0:
                layers.append(nn.Conv2d(num_kernels, num_kernels, kernel_size=kernel_size, stride=conv_stride, padding=padding))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(num_features=num_kernels))
                layers.append(get_activation(activation_fun))
                num_conv = num_conv - 1

        self.conv_layers = nn.Sequential(*layers)

        layers = []
        if num_dense == 1:
            layers.append(nn.Linear(con_out, num_classes))
        else:
            layers.append(nn.Linear(con_out, num_neurons))
            layers.append(get_activation(activation_fun))
        num_dense = num_dense - 1
        while num_dense > 0:
            if num_dense == 1:
                layers.append(nn.Linear(num_neurons, num_classes))
            else:
                layers.append(nn.Linear(num_neurons, num_neurons))
                layers.append(get_activation(activation_fun))
                if dropout:
                    layers.append(nn.Dropout(p=dropout_rate))
            num_dense = num_dense - 1
        self.dense_layers = nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.dense_layers(out)

        return out
def get_data(batch_size=64, split=0.8):
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
    ])

    # Load Fashion MNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    #This is to use only 10% of the data for faster training. Comment it at the end
    train_dataset, rest = random_split(train_dataset, [int(0.1 * len(train_dataset)), int(0.9 * len(train_dataset))])
    test_dataset, rest = random_split(test_dataset, [int(0.1 * len(test_dataset)), int(0.9 * len(test_dataset))])


    train_test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #print(f"train_dataset dataset size: {len(train_dataset)}")
    # Split the training dataset into training and validation sets
    train_size = int(split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print dataset sizes
    #print(f"Training dataset size: {len(train_dataset)}")
    #print(f"Validation dataset size: {len(val_dataset)}")
    #print(f"Test dataset size: {len(test_dataset)}")


    return train_loader, val_loader, test_loader, train_test_loader

def evaluate(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)



    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total
    #f1 = multiclass_f1_score(input, target, num_classes=4)


    true_labels = []
    predicted_labels = []
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class (0 or 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    TP, FP, TN, FN = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[0, 0], conf_matrix[1, 0]

    # Calculate precision and recall
    precision = TP / ((TP + FP) + 0.001)
    recall = TP / ((TP + FN) + 0.001)

    f1_score = 2 * (precision * recall) / ((precision + recall) + 0.001)
    return test_loss, accuracy, f1_score

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, num_classes, optimizer, momentum, l2_pen, l1_norm_rate):
    model.to(device)  # Move the model to the same device as the input data
    model.apply(weights_init)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer_type(optimizer, learning_rate, momentum, model, l2_pen)

    # this is defined to print how many steps are remaining when training
    total_step = len(train_loader)

    lambda1, lambda2 = 0.5, 0.01
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_norm_rate * l1_norm

            optimizer.zero_grad()


            # Backward and optimize
            loss.backward()
            optimizer.step()

            '''
            if (i + 1) % 400 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            '''

def cnn_parameterized(hp):
    num_conv = hp.num_conv
    num_kernels = hp.num_kernels
    kernel_size = hp.kernel_size
    conv_stride = hp.conv_stride
    num_pooling = hp.num_pooling
    pool_size = hp.pool_size
    pool_stride = hp.pool_stride
    num_dense = hp.num_dense
    num_neurons = hp.num_neurons
    padding = hp.padding
    activation_fun = hp.activation_fun
    pool_type = hp.pool_type
    dropout = hp.dropout
    dropout_rate = hp.dropout_rate
    batch_norm = hp.batch_norm
    learning_rate = hp.learning_rate
    epochs = hp.epochs
    batch_size = hp.batch_size
    momentum = hp.momentum
    l1_norm_rate = hp.l1_norm_rate
    optimizer = hp.optimizer
    l2_pen = hp.l2_pen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(num_classes = 10, input_dim = 28, num_conv = num_conv, num_kernels = num_kernels, kernel_size = kernel_size, conv_stride = conv_stride, num_pooling = num_pooling, pool_size = pool_size, pool_stride = pool_stride, num_dense= num_dense, num_neurons= num_neurons, padding=padding, activation_fun=activation_fun, pool_type= pool_type, dropout = dropout, dropout_rate = dropout_rate, batch_norm =batch_norm).to(device)
    print("model", model.valid)

    if model.valid:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_loader, val_loader, test_loader, train_test_loader = get_data(batch_size, 0.8)

        train_model(model, train_test_loader, test_loader, device, num_epochs=epochs, learning_rate=learning_rate, num_classes=10, optimizer=optimizer, momentum=momentum, l2_pen=l2_pen, l1_norm_rate=l1_norm_rate)

        test_loss, test_accuracy, f1_score = evaluate(model, test_loader, nn.CrossEntropyLoss())
        fitness = combined_eval(test_loss, test_accuracy, f1_score)
        print("Fitness of this model:", fitness)
        return fitness
    else:
        print("Not possible model")
        return -1

def combined_eval(loss, accuracy, f1_score):
    if math.isnan(loss) or loss > 1:
        loss = 1
    loss_diff = 1-loss
    return (loss_diff + (accuracy/100) + f1_score) / 3
