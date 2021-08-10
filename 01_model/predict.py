from argparse import ArgumentParser
from PIL import Image
import os
import torch
from torchvision import transforms
from model_repo import *
import matplotlib.pyplot as plt
import json


def main(args):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.isdir('./pics'):
        os.mkdir('./pics')

    pic_path = "./pics/" + args.pics
    img = Image.open(pic_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    if args.model == "GoogLeNet":
        model = eval(args.model)(num_classes=args.numcls, aux_logits=False)
    else:
        model = eval(args.model)(num_classes=args.numcls)

    model_weight_path = './weights/' + args.model + '.pth'
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    plt.title('Inference')
    plt.xlabel(
        "{} - {:.2%}".format(class_indict[str(predict_cla)], predict[predict_cla].item()))
    plt.xticks([])  # 去掉x轴的刻度
    plt.yticks([])  # 去掉y轴的刻度
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser('python3 predict.py')

    # predict config
    parser.add_argument('--model', default='AlexNet', type=str)
    parser.add_argument('--pics', default='1.jpeg', type=str)
    parser.add_argument('--numcls', default=5, type=int)

    main(parser.parse_args())
