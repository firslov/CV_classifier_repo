import os
from typing import Optional

import torch
from torchvision import transforms

from dataproc.dtset import MyDataSet
from dataproc.dtutils import read_split_data, plot_data_loader_image

# http://download.tensorflow.org/example_images/flower_photos.tgz


def custom_dtset(args):
    root, batch_size, pic_size, num_worker, model = args.root, args.bs, args.picsize, args.nw, args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        root)

    # 根据原论文配置不同model的规范化参数
    pic_resize = 256
    pic_size_val = pic_size
    pic_norm_0, pic_norm_1 = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if "efficientnetv2_" in model:
        (pic_size, pic_size_val) = (300, 384) if "_s" in model else (384, 480)
        pic_resize = pic_size
        pic_norm_0, pic_norm_1 = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    elif "efficientnet_" in model:
        index = int(model[-1])
        size_list = [224, 240, 260, 300, 380, 456, 528, 600]
        pic_size = size_list[index]
        pic_size_val = pic_size

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(pic_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(pic_norm_0, pic_norm_1)]),
        "val": transforms.Compose([transforms.Resize(pic_resize),
                                   transforms.CenterCrop(pic_size_val),
                                   transforms.ToTensor(),
                                   transforms.Normalize(pic_norm_0, pic_norm_1)])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    train_num = len(train_data_set)

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    val_num = len(val_data_set)

    # nw = nw if nw else min(
    #     [os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(num_worker))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_worker,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_worker)

    # plot_data_loader_image(train_loader)

    # for step, data in enumerate(train_loader):
    #     images, labels = data

    return train_loader, val_loader, train_num, val_num


if __name__ == '__main__':
    custom_dtset()
