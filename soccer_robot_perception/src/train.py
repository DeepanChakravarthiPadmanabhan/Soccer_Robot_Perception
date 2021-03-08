from torch.nn import CrossEntropyLoss, MSELoss
from model import NimbRoNet2
from torch.utils.data.dataset import ConcatDataset
from utils import read_xml
from torch.optim import SGD, Adam

import torch
import torch.utils.data as data
import random
import glob
import numpy as np
import os
import cv2

from utils import visualize_loss
from PIL import Image

np.set_printoptions(suppress=True)
random.seed(42)

seg_mask = 
seg_img = 

det_mask = 
det_img = 


class DataLoaderSegmentation(data.Dataset):
    def __init__(self, image_paths, target_paths):
        super(DataLoaderSegmentation, self).__init__()
        self.image_paths = glob.glob(image_paths + '*.jpg')
        self.target_paths = glob.glob(target_paths + '*.png')

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        # uncomment next step to remove normalization
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # print('seg image ', np.unique(image))
        mask = cv2.imread(self.target_paths[index])
        mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # print('seg mask ', np.unique(mask))
        # mask = cv2.normalize(mask, None, alpha=0, beta=2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # mask = mask[:, :, 0]

        dim = (224, 224)

        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (56, 56), interpolation=cv2.INTER_AREA)

        # print('seg image shape ', image.shape)
        # print('seg mask shape ', mask.shape)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)


class DataLoaderDetection(data.Dataset):
    def __init__(self, image_paths, target_paths):
        super(DataLoaderDetection, self).__init__()
        self.image_paths = glob.glob(image_paths + '*.png')
        # self.target_paths = glob.glob(target_paths + '*.xml')
        self.target_paths = glob.glob(target_paths + '*.png')

    '''
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        print('det ', image.shape)
        boxes = []
        classes = []

        targets = {}
        for f in self.target_paths:
            classes.append(read_xml(f)[0])
            boxes.append(torch.tensor(read_xml(f)[1]))
            filename.append(read_xml(f)[2])

        targets["boxes"] = boxes
        targets["classes"] = classes
        

        return torch.tensor(image), targets
    '''
    '''
    # this one works 
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        dim = (224, 224)

        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        det_image = np.zeros([56, 56, 3])
        det_image.fill(255)
        classes = []
        boxes = []

        for f in self.target_paths:
            classes.append(read_xml(f)[0])
            boxes.append(torch.tensor(read_xml(f)[1]))

        for obj, box in zip(classes, boxes):
            if obj == 'robot':
                det_image = cv2.circle(det_img,
                                       (int(0.5 * box[2] + 0.5 * box[0]),
                                        int(0.5 * box[3] + 0.5 * box[1])),
                                       5, (0, 255, 255), -1)
            elif obj == 'ball':
                det_image = cv2.circle(det_img,
                                       (int(0.5 * box[2] + 0.5 * box[0]),
                                        int(0.5 * box[3] + 0.5 * box[1])),
                                       5, (255, 255, 0), -1)
            elif obj == 'goalpost':
                det_image = cv2.circle(det_img,
                                       (int(0.5 * box[2] + 0.5 * box[0]),
                                        int(0.5 * box[3] + 0.5 * box[1])),
                                       5, (100, 0, 100), -1)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(det_image, dtype=torch.float32)
    '''

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        # uncomment next step to remove normalization
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mask = cv2.imread(self.target_paths[index])
        # print('det image ', np.unique(image))
        # uncomment next step to remove normalization
        mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # print(np.unique(mask))
        dim = (224, 224)

        # resize image and det masks
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (56, 56), interpolation=cv2.INTER_AREA)

        # print('det image shape ', image.shape)
        # print('det mask shape ', mask.shape)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)


dls = DataLoaderSegmentation(seg_img, seg_mask)
dld = DataLoaderDetection(det_img, det_mask)

concat_dataset = ConcatDataset([dls, dld])

# print('concat ', concat_dataset[0][0].shape)
# print('con ', concat_dataset[0][1].shape)

# print('concat ', concat_dataset[1][0].shape)
# print('con ', concat_dataset[1][1].shape)

# print(concat_dataset[random.randint(0, 138)][0].shape)
# print(concat_dataset[random.randint(0, 138)][1].shape)

# basic dataloader
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                         batch_size=1,
                                         shuffle=True)

model = NimbRoNet2()

criterion_seg = CrossEntropyLoss()  # TO DO: channel specific loss to be implemented in segmentation
criterion_det = MSELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0
    # seg_running_loss = 0.0
    # det_running_loss = 0.0

    # seg_run_loss = []
    # det_run_loss = []

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(inputs)

        unique_elements_in_labels = labels.unique()

        # DETECTION SAMPLE
        if len(unique_elements_in_labels) > 3:
            # print(unique_elements_in_labels)
            inputs = inputs.permute(0, 3, 1, 2)
            # labels = labels.permute(0, 3, 1, 2)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            det_output, seg_output = model(inputs)

            labels = labels.permute(0, 3, 1, 2)

            # print('labels ', labels.shape)
            # print('det ', det_output.shape)

            loss = criterion_det(det_output, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

        # SEGMENTATION SAMPLE
        elif len(unique_elements_in_labels) < 4:
            # print(unique_elements_in_labels)
            inputs = inputs.permute(0, 3, 1, 2)
            # labels = labels.permute(0, 3, 1, 2)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            det_output, seg_output = model(inputs)

            # print('seg_output shape ', seg_output.shape)
            # print('labels shape ', labels.shape)
            labels = labels[:, :, :, 0]
            # print('labels shape ', labels.shape)
            loss = criterion_seg(seg_output, labels.long())

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 2 == 0:  # print every 2 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

torch.save(model, )
# visualize_loss(running_loss)
# print(running_loss)

print('Finished Training')

# references
# https://discuss.pytorch.org/t/how-make-customised-dataset-for-semantic-segmentation/30881/5
# https://towardsdatascience.com/unbalanced-data-loading-for-multi-task-learning-in-pytorch-e030ad5033b
# https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
