import torch
import os
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss, MSELoss

random.seed(42)


def read_xml(path):
    objects_in_image = []
    xmin, ymin, xmax, ymax = [], [], [], []
    tree = ET.parse(path)
    root = tree.getroot()  # get root node
    desired_node = root.findall('object')  # get all object nodes
    filename = root.findall('filename')[0].text

    for obj in range(len(desired_node)):  # parse through the objects present
        objects_in_image.append(desired_node[obj].find('name').text)

        xmin.append(int(desired_node[obj].find('bndbox').find('xmin').text))  # extracting the bounding box coordinates
        ymin.append(int(desired_node[obj].find('bndbox').find('ymin').text))
        xmax.append(int(desired_node[obj].find('bndbox').find('xmax').text))
        ymax.append(int(desired_node[obj].find('bndbox').find('ymax').text))

    xmin = np.asarray(xmin).reshape((-1, 1))  # converting the list to numpy array
    ymin = np.asarray(ymin).reshape((-1, 1))
    xmax = np.asarray(xmax).reshape((-1, 1))
    ymax = np.asarray(ymax).reshape((-1, 1))

    bndbox = np.hstack((xmin, ymin, xmax, ymax))

    return objects_in_image, bndbox, filename


def visualize_bbox(path, img, bbox, classes):
    im = cv2.imread(path + img + '.png')
    for i in range(bbox.shape[0]):
        if classes[i] == 'robot':  # draw a yellow box for robot
            cv2.rectangle(im, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]),
                          (0, 255, 255), 3)

        elif classes[i] == 'ball':  # draw a cyan box for ball
            cv2.rectangle(im, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]),
                          (255, 255, 0), 3)

        elif classes[i] == 'goalpost':  # draw a magenta box for goalpost
            cv2.rectangle(im, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]),
                          (100, 0, 100), 3)

    cv2.imshow('im', im)
    cv2.waitKey(0)


def convert_bbox_coordinates_to_colored_blobs(path, img, bbox, classes):
    img_name = img.split('.')[0]
    print('IMG_NAME ', img_name)
    print(path + '/' + img[0] + '.png')
    im = cv2.imread(path + '/' + img[0] + '.png')

    blob_image = np.zeros([im.shape[0], im.shape[1], 3])  # create a white background image to draw blobs on
    blob_image.fill(255)

    for box in range(bbox.shape[0]):
        # bbox coordinate format (xmin, ymin, xmax, ymax)
        if classes[box] == 'robot':  # draw a yellow blob for robot (centre point of base line of bounding box)
            img = cv2.circle(blob_image,
                             (int(0.5 * bbox[box][2] + 0.5 * bbox[box][0]),
                              int(bbox[box][3])),
                             5, (0, 255, 255), -1)
        elif classes[box] == 'ball':  # draw a cyan blob for ball (centre point of the ball)
            img = cv2.circle(blob_image,
                             (int(0.5 * bbox[box][2] + 0.5 * bbox[box][0]),
                              int(0.5 * bbox[box][3] + 0.5 * bbox[box][1])),
                             5, (255, 255, 0), -1)
        elif classes[box] == 'goalpost':  # draw a magenta blob for goalpost (centre point of base line of bounding box)
            img = cv2.circle(blob_image,
                             (int(0.5 * bbox[box][2] + 0.5 * bbox[box][0]),
                              int(bbox[box][3])),
                             5, (100, 0, 100), -1)

        new_name = os.path.join(os.path.join(os.path.dirname(path), 'masks'), img_name) + '.png'
    print(new_name)
    cv2.imwrite(new_name, img)


def convert_mask_color_pixels_to_class_ids(path):
    # convert the RGB annotation to grayscale mask
    for file in os.listdir(path):
        im = cv2.imread(path + '/' + file)
        mskIm = np.zeros((im.shape[0], im.shape[1], 3), dtype=int)
        for row in range(im.shape[0]):
            for col in range(im.shape[1]):
                if (im[row][col] == np.array([0, 0, 0])).all():  # assign class id = 0 for background
                    mskIm[row][col][:] = 0
                elif (im[row][col] == np.array([0, 128, 128])).all():  # assign class id = 1 for field
                    mskIm[row][col][:] = 2
                elif (im[row][col] == np.array([0, 128, 0])).all():  # assign class id = 2 for lines
                    mskIm[row][col][:] = 1

        new_name = os.path.dirname(path) + '/masks/' + file
        print(new_name)
        cv2.imwrite(new_name, mskIm)


def plot_output(tens, txt, dim):
    out = tens.permute(2, 3, 1, 0)
    out = out.squeeze()
    # out = torch.abs(out)

    out = out.detach().numpy()
    out = cv2.resize(out, dim, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(out, (5, 5), 0)  # gaussian filter

    print(blur.shape)
    print(blur)
    cv2.imshow(txt + '-' + str(dim), blur)
    cv2.imshow(txt+'-'+str(dim)+'0', blur[:, :, 0])
    cv2.imshow(txt+'-'+str(dim)+'1', blur[:, :, 1])
    cv2.imshow(txt+'-'+str(dim)+'2', blur[:, :, 2])
    cv2.waitKey(0)


def visualize_loss(loss):
    plt.plot(range(len(loss)), loss)
    plt.show()


# for understanding cross entropy
def cross_entropy(prediction, label_pth):
    im = cv2.imread(label_pth)
    im = torch.from_numpy(im)
    loss = CrossEntropyLoss()

    print('prediction dimension ', prediction.shape)

    im = im.unsqueeze_(0)
    im = im[:, :, :, 0]
    print('target final dimension ', im.shape)

    # print(np.unique(prediction))
    print(prediction)

    # print(np.unique(im))
    print(im)
    print(loss(prediction, im))


def read_img(path):
    for file in os.listdir(path):
        print(path+'/'+file)
        img = cv2.imread(path+'/'+file)
        print(img.shape)
        print(img[:, :, 0].shape)

        cv2.imshow(file, img)
        '''
        cv2.imshow('im_fc', img[:, :, 0])
        cv2.imshow('im_sc', img[:, :, 1])
        cv2.imshow('im_tc', img[:, :, 2])
        '''
        cv2.waitKey(0)


def generateGaussian(bbox, object, var):
    dir_gauss = cv2.getGaussianKernel(100, 10.0)
    print(dir_gauss.shape)
    kernel = 255*np.multiply(dir_gauss.T, dir_gauss)
    print("Kernel: \n", kernel)

    cv2.imshow('k', kernel)
    cv2.waitKey(0)
    return dir_gauss



# References:
# https://www.datacamp.com/community/tutorials/python-xml-elementtree
