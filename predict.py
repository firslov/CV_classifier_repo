from argparse import ArgumentParser
from PIL import Image
import os
import torch
from torchvision import transforms
from model_repo import *
import matplotlib.pyplot as plt
import json


def main(args):

    pic_resize = 256
    pic_size = args.picsize
    pic_norm_0, pic_norm_1 = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if "efficientnetv2_" in args.model:
        pic_size = 384 if "_s" in args.model else 480
        pic_resize = pic_size
        pic_norm_0, pic_norm_1 = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif "efficientnet_" in args.model:
        index = int(args.model[-1])
        size_list = [224, 240, 260, 300, 380, 456, 528, 600]
        pic_size = size_list[index]
        pic_resize = pic_size

    data_transform = transforms.Compose([transforms.Resize(pic_resize),
                                         transforms.CenterCrop(pic_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(pic_norm_0, pic_norm_1)])

    if not os.path.isdir('./test_pics'):
        os.mkdir('./test_pics')

    pic_path = "./test_pics/" + args.pics
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
    parser.add_argument('--picsize', default=224, type=int)

    main(parser.parse_args())
