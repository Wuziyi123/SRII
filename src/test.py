import torch
from torch.utils.data import DataLoader
from ResNet import resnet18_cbam

import os
from iCaRL import iCaRLmodel


def get_train_and_test_dataloader(model, classes):
    if classes[0] == 0:
        model.exemplar_set = []
    model.train_dataset.getTrainData(classes, model.exemplar_set)

    model.test_dataset.getTestData(classes)

    train_loader = DataLoader(dataset=model.train_dataset,
                              shuffle=True,
                              batch_size=128)

    test_loader = DataLoader(dataset=model.test_dataset,
                             shuffle=True,
                             batch_size=128)

    return train_loader, test_loader


def test(parser_data):

    os.chdir(r'../')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    numclass = 20
    feature_extractor = resnet18_cbam()
    img_size = 32
    task_size = 10
    memory_size = 2000
    epochs = 100
    batch_size = parser_data.batch_size
    learning_rate = parser_data.learning_rate
    test_num = parser_data.test_classes
    test_path = parser_data.test_file_path
    data_path = 'dataset'

    load = iCaRLmodel(data_path, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
    model = load.model

    if os.path.isfile(test_path):
        print("=> loading checkpoint '{}'".format(test_path))
        checkpoint = torch.load(test_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(test_path))

    model.to(device)

    classes = [0, test_num]  # at here, we test 20 classes(1 base step + 1 incremental step)
    _, test_loader = get_train_and_test_dataloader(load, classes)

    model.eval()
    correct, total = 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)  # outputs = (128,20)

        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    print(str(accuracy.item()))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # learning rate
    parser.add_argument('--learning_rate', default=2.0, type=float, help='learning rate')
    # num of classes for test
    parser.add_argument('--test_classes', default=20, type=int, help='num of classes for test')
    # Root directory of training dataset
    parser.add_argument('--test_file_path', default='model/test-20classes.pth.tar', type=str, help='path of test file', metavar='PATH_TO_TEST_FILE')
    # batch size
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    print(args)
    test(args)




