import pandas as pd
from matplotlib import pyplot as plt
import random
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from read_roi import read_roi_zip
import cv2

def load_data(file_name='/home/silasi/Bead Test Data/5 - Data/73594-3 - Manual Bead Location Data v0.1.4 - Dilation Factor 2.csv'):

    if os.path.exists(file_name):
        data = pd.read_csv(file_name)
        data_frame = []
        for i in range(data.values.shape[0]):
            values = data.values[i]
            mean = values[1]
            x = float(values[2])
            y = float(values[3])
            z = "None"
            if values[4] != " Flat":
                z = float(values[4])
                data_frame.append([x, y, z, mean])


    else:
        print("File not found! : %s"%file_name)
        raise IOError

    return data_frame

class Bead:
    def __init__(self, pos_xy=(), pos_z=None, brightness=0):
        self.pos = [pos_xy]
        self.start_z = pos_z
        self.end_z = pos_z
        self.brightness = brightness
        self.disconnected_number = 0
        self.closed = False

def find_real_bead(data_frame, deliation_factor=3, pixel2mm = 0.005464, tolerance=0.5, ignore_disconnected=3):
    layers = []
    current_z_pos = data_frame[0][2]
    layer = []
    bead_list = []

    deliation_distance = pixel2mm * 2 * (deliation_factor ** 2)

    def manhattan_key(elem):
        return elem[0] + elem[1]
    def distance_key(elem):
        return -elem[4]

    for item in data_frame:
        if item[2] != current_z_pos:
            layers.append(layer)
            layer = []
            current_z_pos = item[2]
        if item[2] == current_z_pos:
            layer.append(item)
    layers.append(layer)
    for i in range(len(layers)):
        layer = layers[i]
        layer.sort(key=manhattan_key)

        if len(bead_list) == 0:
            for item in layer:
                bead = Bead((item[0], item[1]), item[2], item[3])
                bead_list.append(bead)
        else:
            for bead in bead_list:
                if not bead.closed:
                    for i in range(len(layer)):
                        bead_x = bead.pos[-1][0]
                        bead_y = bead.pos[-1][1]
                        distance = (bead_x - layer[i][0]) ** 2 + (bead_y - layer[i][1]) ** 2
                        if distance < deliation_distance:
                            if len(layer[i]) < 5:
                                layer[i].append(distance)
                            else:
                                layer[i][4] = distance
                        else:
                            if len(layer[i]) < 5:
                                layer[i].append(-1000)
                            else:
                                layer[i][4] = -1000
                    layer.sort(key=distance_key)
                    if len(layer) == 0:
                        bead.disconnected_number += 1
                        if bead.disconnected_number >= ignore_disconnected:
                            bead.closed = True
                    elif layer[0][4] != -1000 and float(layer[0][3]) * tolerance < float(bead.brightness):
                        bead.pos.append((layer[0][0], layer[0][1]))
                        bead.end_z = layer[0][2]
                        bead.brightness = layer[0][3]
                        layer.pop(0)
                    else:
                        bead.disconnected_number += 1
                        if bead.disconnected_number >= ignore_disconnected:
                            bead.closed = True
            for item in layer:
                bead = Bead((item[0], item[1]), item[2], item[3])
                bead_list.append(bead)

    return bead_list

class Point:
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.color = []

def plot_beads(bead_list, number_to_show=False, show_real_bead=False):
    point_list = []

    for bead in bead_list:
        color = random.randrange(0, 100)
        point = Point()
        if show_real_bead:
            depth_2_show = 1
        else:
            depth_2_show = len(bead.pos)
        for i in range(depth_2_show):
            x = bead.pos[i][0]
            y = bead.pos[i][1]
            z = bead.start_z + float(i) * 0.05
            point.x.append(x)
            point.y.append(y)
            point.z.append(z)
            point.color.append(color)
        point_list.append(point)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not number_to_show:
        number_to_show = len(point_list)
    for point in point_list[0:number_to_show]:
        ax.scatter(point.x, point.y, point.z, depthshade=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def label_unique_points(img_dir, bead_list):
    def z_key(elem):
        return float(elem.split(',')[-1].strip().split('.tif')[0])

    tif_list = os.listdir(img_dir)
    tif_list.sort(key=z_key)

    z_list = []
    for tif in tif_list:
        z_list.append(z_key(tif))

    layers = find_unique_point_in_layers(bead_list, z_list)

    origin = calculate_origin(img_dir)
    index = 0
    while True:
        img = cv2.imread(os.path.join(img_dir, tif_list[index]))
        layer = layers[index]
        if len(layer) > 0:
            for point in layer:
                mmpoint = (point[0], point[1])
                pixel_point = mm2pixel(mmpoint, origin)
                cv2.circle(img, pixel_point, 15, (255, 0, 0), thickness=5)
        img = cv2.resize(img, origin)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'index: %d'%index, (50, 80), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, '<- : previous || -> : next || q : exit', (50, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('View of Diconnected Click', img)
        key = cv2.waitKey()
        if key == 81:
            index -= 1
            if index < 0:
                index = len(tif_list) - 1
        elif key == 83:
            index += 1
            if index >= len(tif_list):
                index = 0
        elif key == 113:
            break

def calculate_origin(img_dir):
    tif_list = os.listdir(img_dir)

    for tif_name in tif_list:
        last_temp = tif_name.split(',')[-1].strip().split('.tif')[0]
        if float(last_temp) == 0:
            center_section = os.path.join(img_dir, tif_name)
            break

    img = cv2.imread(center_section)

    centre_point = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))

    return centre_point


def find_unique_point_in_layers(bead_list, z_list):
    def depth_key(elem):
        return elem.end_z
    bead_list.sort(key=depth_key)
    layers = []
    for z in z_list:
        layers.append([])
        for bead in bead_list:
            if (bead.start_z == bead.end_z) and (bead.end_z == z):
                layers[-1].append(bead.pos[0])
    return layers


def mm2pixel(point, centre_point, pixel2mm=0.005464):
    pixel_x = int(-float(point[0])/pixel2mm + centre_point[0])
    pixel_y = int(-float(point[1])/pixel2mm + centre_point[1])
    return (pixel_x, pixel_y)


def main(rootDir):

    csv_dir = os.path.join(rootDir, "5 - Data", "73594-2 - Manual Bead Location Data v0.1.4 - Dilation Factor 2.csv")
    img_dir = os.path.join(rootDir, "3 - Processed Images", "7 - Counted Reoriented Stacks Renamed")

    data_frame = load_data(csv_dir)
    bead_list = find_real_bead(data_frame, deliation_factor=3, ignore_disconnected=1, tolerance=0.3)
    plot_beads(bead_list, False, True)
    label_unique_points(img_dir, bead_list)



if __name__ == '__main__':
    rootDir = "/home/silasi/73594-2"
    main(rootDir)


