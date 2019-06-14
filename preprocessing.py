'''
Adjust the position of the images based on a point clicked bu user
'''
import cv2
import os
import numpy as np

click_point = (0, 0)

def mouse_click(event, x, y, flags, param):
    '''
    Use for getting click points in real images
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    '''
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point  = (x, y)

def show_img(img, introduction=True):
    '''
    display the img in a proper size
    :param img:
    :return:
    '''
    img_show = img.copy()
    img_show = cv2.resize(img_show, (int(img_show.shape[1] * 0.5), int(img_show.shape[0] * 0.5)))
    if introduction:
        cv2.putText(img_show, 'e next frame', (50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255))
        cv2.putText(img_show, 'q previous frame', (50, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 's confirm the point you clicked', (50, 150), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
    cv2.circle(img_show, click_point, 10, (255, 255, 255), 3, 8)
    cv2.imshow('calibration', img_show)
    key = cv2.waitKey(1) & 0xFF
    return key

def get_img_list(name="61148-3", root_dir="/home/silasi/brain_imgs/raw_images"):
    '''
    get a list of images from image folder
    :param name:
    :param root_dir:
    :return: a list of opencv image instances
    '''
    img_dir_list = [os.path.join(root_dir, name, i) for i in os.listdir(os.path.join(root_dir, name))]
    def sort_key(elem):
        return -int(elem.split('aligned')[-1].split('.tif')[0])
    img_dir_list.sort(key=sort_key)
    return [cv2.imread(dir) for dir in img_dir_list]

def mkdir(name, root_dir="/home/silasi/brain_imgs/processed_images"):
    '''
    make the proper file structure for calibrated images
    :param name:
    :param root_dir:
    :return:
    '''
    root_dir = os.path.join(root_dir, name)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    processed_dir = os.path.join(root_dir, "3 - Processed Images")
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    counted_dir = os.path.join(processed_dir, "7 - Counted Reoriented Stacks Renamed")
    if not os.path.exists(counted_dir):
        os.mkdir(counted_dir)

    data_dir = os.path.join(root_dir, "5 - Data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return counted_dir

def affin_image(img):
    '''
    shift the image by the point user clicked
    :param img:
    :return:
    '''
    global click_point

    img_center = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    shift_col = img_center[0] - click_point[0] * 2
    shift_row = img_center[1] - click_point[1] * 2
    M = np.float32([[1, 0, shift_col], [0, 1, shift_row]])
    canvas = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return canvas

def main(name="61148-3", root_dir="/home/silasi/brain_imgs/raw_images", target_dir="/home/silasi/brain_imgs/processed_images"):
    '''
    Transformed the sequence of brain[name] images under root_dir folder, the processed images will be saved under target_dir folder
    :param name:
    :param root_dir:
    :param target_dir:
    :return:
    '''
    img_list = get_img_list(name, root_dir)
    index = 0
    cv2.namedWindow('calibration')
    cv2.setMouseCallback('calibration', mouse_click)
    while(True):
        key = show_img(img_list[index])
        if key == 113:
            index += 1
            if index >= len(img_list):
                index = 0
        elif key == 101:
            index -= 1
            if index < 0:
                index = len(img_list) - 1
        elif key == 115:
            break

    counted_dir = mkdir(name, target_dir)
    for i in range(len(img_list)):
        temp = "%s, %03d, %.2f.tif"%(name, i, 0.05 * (i- index))

        img = affin_image(img_list[i])
        cv2.imwrite(os.path.join(counted_dir, temp), img)

    print("finished")

if __name__ == '__main__':
    name_list = ["61148-3", "61149-1", "61149-2", "66148-1", "stack"]
    for name in name_list:
        main(name)