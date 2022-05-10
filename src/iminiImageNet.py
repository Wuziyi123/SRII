from ImageNet100 import *
import numpy as np
from PIL import Image


class iminiImageNet(ImageNet100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None):
        super(iminiImageNet, self).__init__(root, train=train)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.transform = transform
        self.target_transform = target_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.incremental = False
        self.up_model_TrainData = []
        self.up_model_TrainLabels = []
        # self.img_paths = [os.path.join("mini-imagenet/images", i) for i in self.data]

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        #  con_data = ndarray
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas,labels=self.concatenate(datas,labels)
        self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
        self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
        print("the size of test set is %s"%(str(self.TestData.shape)))
        print("the size of test label is %s"%str(self.TestLabels.shape))


    def getBranchTestData(self, classes):
        datas,labels=[],[]
        self.TestData = []
        self.TestLabels = []
        for label in range(0, classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas,labels=self.concatenate(datas,labels)
        self.TestData=datas
        self.TestLabels=labels
        print("the size of test set is %s"%(str(self.TestData.shape)))
        print("the size of test label is %s"%str(self.TestLabels.shape))


    def getTrainData(self,classes,exemplar_set):
        datas,labels=[],[]
        if len(exemplar_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length),label) for label in range(len(exemplar_set))]

        if classes[0] >= 0:
            for label in range(classes[0], classes[1]):
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
        # datas = [(500,32,32,3),(500,32,32,3)...]   labels = [(500,),(500,)...]
        # con_data=np.concatenate((con_data,datas[i]),axis=0)
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))


    def getDoubleBranchTrainData(self,up_model,classes, i):

        self.up_model_TrainData, self.up_model_TrainLabels = up_model.train_dataset.TrainData,\
                                                 up_model.train_dataset.TrainLabels
        add_len = up_model.memory_size
        datas, labels = [], []

        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label-10*i))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        rehearsal_Size = len(self.up_model_TrainData)-len(self.TrainData)

        if add_len:
            self.TrainData = np.concatenate((self.TrainData[:rehearsal_Size], self.TrainData), axis=0)
            self.TrainLabels = np.concatenate((self.TrainLabels[:rehearsal_Size], self.TrainLabels), axis=0)

        print("the size of train set is %s"%(str(self.TrainData.shape)))
        print("the size of train label is %s"%str(self.TrainLabels.shape))

    def getDoubleBranchTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        up_img, up_target = Image.fromarray(self.up_model_TrainData[index]), self.up_model_TrainLabels[index]

        if self.transform:
            img = self.transform(img)
            up_img = self.transform(up_img)

        if self.target_transform:
            target = self.target_transform(target)
            up_target = self.target_transform(up_target)

        return index, img, target, up_img, up_target


    def getTrainItem(self,index):
        # c = self.TrainData[index]
        # img = np.array(Image.open(self.img_paths[index]))
        # img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        img, target = Image.open(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index, img, target


    def getTestItem(self,index):
        # img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img, target = Image.open(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if not self.incremental:
            if list(self.TrainData) !=[]:
                return self.getTrainItem(index)
            elif list(self.TestData) !=[]:
                return self.getTestItem(index)
        else:
            if list(self.TrainData) !=[]:
                return self.getDoubleBranchTrainItem(index)
            elif list(self.TestData) !=[]:
                return self.getTestItem(index)


    def __len__(self):
        if list(self.TrainData) !=[]:
            return len(self.TrainData)
        elif list(self.TestData) !=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        # images = self.data[np.array(self.targets) == label]
        images_path = self.data[np.array(self.targets) == label]
        img = []
        for i in range(len(images_path)):
            add = np.array(Image.open(images_path[i]))
            img.append(add)
        img = np.array(img, dtype=object)
        return img, images_path
