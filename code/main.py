from baseline import *
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.0')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='input batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--save', type=bool, default=True,
                        help='save model (default: True)')
    args = parser.parse_args()

    # Codes
    train_loader, test_loader, categories = init(args.batch_size)
    visualize(train_loader, categories)
    net = return_baseline()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    summary(net, (3, 512, 512))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    train(num_epochs=args.epochs, loader=train_loader, device=device, net=net,
          criterion=criterion, optimizer=optimizer)
    inference(loader=test_loader, device=device, net=net)
    torch.save(net, str(time.time()) + '.pt')
