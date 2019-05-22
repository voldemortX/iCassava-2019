import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time


# Baseline: ResNet-18
def return_baseline():
    net = torchvision.models.resnet18(pretrained=False)
    return net


# Draw images
def show(images):
    images = images / 2 + 0.5  # denormalize
    np_images = images.numpy()
    print(np_images.shape)
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    plt.show()


# Show random images
def visualize(loader, categories):
    temp = iter(loader)
    images, labels = temp.next()
    show(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % categories[labels[j]] for j in range(labels.size(0))))


# Load data
def init(batch_size):
    transform = transforms.Compose(
        [transforms.Resize((512, 512)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # !!! Order matters

    train_set = torchvision.datasets.ImageFolder(root="../data/train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    dev_set = torchvision.datasets.ImageFolder(root="../data/dev", transform=transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    categories = ('cbb', 'cbsd', 'cgm', 'cmd', 'healthy')

    return train_loader, dev_loader, categories


# Train data
def train(num_epochs, loader, device, optimizer, criterion, net):
    for epoch in range(num_epochs):
        running_loss = 0.0
        time_now = time.time()
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        print('Epoch time: %.2fs' % (time.time() - time_now))


# Test
def inference(loader, device, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test acc: %f' % (100 * correct / total))
