from argparse import ArgumentParser
import torch.optim as optim
import torch.nn as nn
import torch
from model_repo import *
import dataproc
import os
import time


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_num, val_num = dataproc.dtcustom.custom_dtset(
        args.root, args.bs, args.picsize, args.nw, args.model)

    net = eval(args.model)(num_classes=args.numcls)
    # pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}

    if not os.path.isdir('./weights'):
        os.mkdir('./weights')

    pre_weights = './weights/' + args.model + \
        '.pth' if args.pre == "self" else args.pre

    if args.pre:
        net.load_state_dict(torch.load(pre_weights))
        print("Weight is loaded.")

    # for param in net.features.parameters():
    #     param.requires_grad = False

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    save_path = './weights/' + args.model + '.pth'
    best_acc = 0.0

    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        t = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            output = net(images.to(device))
            loss = loss_function(output, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print(
                "\rTrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print("Time cost: {:.2f}".format(time.perf_counter() - t))

        net.eval()
        acc = 0.0
        with torch.no_grad():
            print("Testing...")
            for data_test in val_loader:
                test_images, test_labels = data_test
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == test_labels.to(device)).sum().item()
            accurate_test = acc / val_num
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(net.state_dict(), save_path)
            print('[Epoch %d] Train_loss: %.3f  Test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, acc / val_num))

    print('Finished Training.')


if __name__ == '__main__':
    parser = ArgumentParser('python3 train.py')

    # training config
    parser.add_argument('--model', default='AlexNet', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batchSize', dest='bs', default=8, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--pre', default='', type=str)
    parser.add_argument('--numWorkers', dest='nw', default=0, type=int)
    parser.add_argument('--picsize', default=224, type=int)

    # dataset
    parser.add_argument('--dataDir', dest='root',
                        default="./dataset/flower_data/flower_photos", type=str)
    parser.add_argument('--numcls', default=5, type=int)

    main(parser.parse_args())
