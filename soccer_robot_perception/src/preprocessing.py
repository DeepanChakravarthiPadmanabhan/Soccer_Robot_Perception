import os
import argparse

from utils import read_xml, visualize_bbox, convert_bbox_coordinates_to_colored_blobs
from utils import convert_mask_color_pixels_to_class_ids


def main():
    parser = argparse.ArgumentParser(description='Visualize the ground truth')

    parser.add_argument('--data_folder', type=str, metavar='D',
                        help='Folder where data is located')
    parser.add_argument('--task', type=str, metavar='T',
                        help='task to perform')

    args = parser.parse_args()

    path = os.path.join(args.data_folder, args.task)
    # print(path)

    annotation_path = os.path.join(path, 'annotations')
    # print(annotation_path)

    image_path = os.path.join(path, 'images')
    # print(image_path)

    if args.task == 'segmentation':
        if len(os.listdir(os.path.join(path, 'masks'))) != 0:
            print('Masks already generated')
        else:
            convert_mask_color_pixels_to_class_ids(annotation_path)

    elif args.task == 'detection':
        # parsing through all xml files
        for file in os.listdir(annotation_path):
            if '.xml' in file:
                object_list, bounding_box, image_name = read_xml(annotation_path + '/' + file)
                # visualize_bbox(args.data_folder, image_name, bounding_box, object_list)  # plot the bbox on the object
                convert_bbox_coordinates_to_colored_blobs(image_path, file, bounding_box, object_list)  # plot the blobs on the object


if __name__ == '__main__':
    main()
