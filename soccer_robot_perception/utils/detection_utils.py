import xml.etree.ElementTree as ET


def read_xml_file(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bb_list = []
    class_list = []

    for boxes in root.iter("object"):
        filename = root.find("filename").text
        class_list.append(boxes.find("name").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        bb_list.append([xmin, ymin, xmax, ymax])

    return class_list, bb_list
