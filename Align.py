import nrrd
import numpy as np
import pickle as pk
from skimage.transform import PiecewiseAffineTransform, warp
from tqdm import tqdm
import skimage
from bead_finder import *

# The position of the original point in NRRD file.
NRRD_CERTER_POSITION = (104, 106, 114)

# Used for opencv click function.
click_point_nrrd = None
click_point_real = None



def load_reference_data():
    '''
    Retrun the nrrd file
    :return:
    '''
    nrrd_directory = "nrrd_reference/annotation_50.nrrd"
    query_directory = "nrrd_reference/query.csv"
    data, header = nrrd.read(nrrd_directory)
    df = pd.read_csv(query_directory)
    return data, df


def nrrd2ImageList(data, save_directory="nrrd_reference", show=True):
    '''
    Transfer the nrrd file into a pickled image list, and also save a dictionary which project the pixel value to labels in nrrd.
    :param data:
    :param save_directory:
    :return:
    '''
    grey_dict = {}
    value = 0
    (depth, height, length) = data.shape
    canvas_list = []
    for index in range(depth):
        canvas = np.zeros((height, length)).astype(np.uint8)
        for i in range(height):
            for j in range(length):
                if data[index][i][j] != 0:
                    if data[index][i][j] not in grey_dict.keys():
                        grey_dict[data[index][i][j]] = int( (value * 0.70 / 670. + 0.12) * 255.)
                        value += 1
                    canvas[i][j] = grey_dict[data[index][i][j]]
        canvas_list.append(canvas)
        if show:
            cv2.imshow('1', canvas)
            cv2.waitKey()

    f1 = open(os.path.join(save_directory, 'nrrd_plot.pickle'), 'wb')
    pk.dump(canvas_list, f1, protocol=pk.HIGHEST_PROTOCOL)
    f2 = open(os.path.join(save_directory, 'nrrd_dict.pickle'), 'wb')
    pk.dump(grey_dict, f2, protocol=pk.HIGHEST_PROTOCOL)

def get_adaptive_threshold(img_gray, show=False):
    '''
    Using the hist diagram to calculate the adaptive threshold of binarizing th image
    :param img_gray: single channel gray image
    :param show: if show is true, it will open a window containing the hist diagram
    :return: Adapative threshold value
    '''
    hist_full = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    if show:
        plt.plot(hist_full)
        plt.show()
    hist_ave = sum(hist_full[1:]) / 255.
    window_size = 5
    for i in range(10, 50):
        temp = hist_full[i: i + window_size].reshape((window_size  , ))
        if np.gradient(temp).max() < 0 and (temp.sum() / float(window_size)) < hist_ave:
            return i
    return 25


def is_in_center(centerPonit, real_img):
    '''
    Given a center point and the ima containing the brain, determine whether the region is in the cernter of image or not
    :param centerPonit:
    :param real_img:
    :return: boolean
    '''
    x = float(real_img.shape[1] * 0.5)
    y = float(real_img.shape[0] * 0.5)
    if (centerPonit[0] > (x * 0.6)) and (centerPonit[0] < (x * 1.4)) and (centerPonit[1] > (y * 0.4)) and (centerPonit[1] < (y * 1.4)):
        return True
    else:
        return False

def get_convex_hull(real_img, show_real=False, show_binary=False, constant_threshold=None, approximate=True):
    '''
    Calculatet the convex hull (contours) of the brain in the image
    :param real_img: The image for brain tissue
    :return: convex hull the approximating  contours
    '''
    if len(real_img.shape) == 3:
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    if constant_threshold:
        threshold = constant_threshold
        ret, th = cv2.threshold(real_img, threshold, 255, cv2.THRESH_BINARY)
    else:
        threshold = get_adaptive_threshold(real_img)
        ret, th = cv2.threshold(real_img, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        th = cv2.erode(th, kernel, iterations=2)


    if show_binary:
        real_img_2_show = th.copy()
        img_2_show = cv2.resize(real_img_2_show, (int(real_img_2_show.shape[1] * 0.5), int(real_img_2_show.shape[0] * 0.5)))
        cv2.imshow('????', img_2_show)
        cv2.waitKey()

    _, contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_2_show = real_img.copy()
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        center = (int(x + w * 0.5), int(y + h * 0.5))
        if cv2.contourArea(contours[i]) > 2e5:
            if is_in_center(center, real_img):
                if approximate:
                    hull = cv2.convexHull(contours[i])
                else:
                    hull = contours[i]
                if show_real:
                    cv2.rectangle(img_2_show, (x, y), (x + w, y + h), (255, 255, 0), 5)
                    cv2.drawContours(img_2_show, [hull], 0, (255, 255, 255), 1, 8)
                    img_2_show = cv2.resize(img_2_show, (int(img_2_show.shape[1] * 0.5), int(img_2_show.shape[0] * 0.5)))
                    cv2.imshow('1', img_2_show)
                    cv2.waitKey()
                return hull
    return -1

def get_blank_canvas(img_real):
    '''
    Get a empty black canvas with the same shape as img_real
    :param img_real:
    :return:
    '''
    canvas = np.zeros((img_real.shape[0], img_real.shape[1])).astype(np.uint8)
    return canvas

def get_radioactive_lines(center, length=1300.):
    '''
    get the end point of radioatcive lines from a given center point
    :param center:
    :param length:
    :return:
    '''
    angle = np.linspace(0, 360, 60).astype(np.int).tolist()
    point_list = []
    for i in angle:
        x = int(center[0] + np.cos(i * np.pi / 180.) * length)
        y = int(center[1] + np.sin(i * np.pi / 180.) * length)
        point_list.append((x, y))
    return point_list

def calculate_intersection_point(hull, img_real, show=False):
    '''
    Get the feature points from the contours of brain in the img_real
    :param hull: The convex hull
    :param img_real: The image for real tissue
    :param show: If show, then will show the middle result
    :return:
    '''

    center_point = (int(img_real.shape[1] * 0.5), int(img_real.shape[0] * 0.5))
    line_list = get_radioactive_lines(center_point)

    canvas = get_blank_canvas(img_real)
    point_list = []
    for p1 in line_list:
        p2 = center_point
        result = createLineIterator(p2, p1, canvas)
        dis_list = []
        for i in result:
            dis = cv2.pointPolygonTest(hull, (i[0], i[1]), True)
            dis_list.append(abs(dis))
            if show:
                cv2.line(canvas, center_point, (i[0], i[1]), (255, 255, 255), 1, 8)
        index = dis_list.index(min(dis_list))
        point = (result[index][0], result[index][1])
        point_list.append(point)
        if show:
            cv2.circle(canvas, point, 10, (255, 255, 255), 1, 8)
    if show:
        cv2.drawContours(canvas, [hull], 0, (255, 255, 255), 1, 8)
        show_img(canvas)
    return  point_list

def createLineIterator(P1, P2, img):
    '''
    Get all the points from the line (p1, p2)
    :param P1:
    :param P2:
    :param img:
    :return:
    '''
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

def show_img(img, introduction=False):
    '''
    Show the image in a proper size
    :param img: The image to show
    :param introduction: if true, will display instructions on image
    :return:
    '''
    img_show = img.copy()
    img_show = cv2.resize(img_show, (int(img_show.shape[1] * 0.5), int(img_show.shape[0] * 0.5)))
    if introduction:
        cv2.putText(img_show, 'q: reduce weight', (50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 'e: increase weight', (50, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 'a: previous frame', (50, 150), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 'd: next frame', (50, 200), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 's: quit', (50, 250), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
    cv2.imshow('img_show', img_show)
    key = cv2.waitKey()
    return key

def get_pure_brain_nrrd(nrrd_frame, refactored_nrrd_center):
    '''
    Return only the brain part in the nrrd frame, and remove the outside black area
    :param nrrd_frame:
    :param refactored_nrrd_center:
    :return:
    '''
    (height, width) = nrrd_frame.shape
    row1 = 0
    col1 = 0
    row2 = 0
    col2 = 0
    for row in range(0, height):
        if np.asarray(nrrd_frame[row, :]).sum() > 0:
            row1 = row
            break

    for row in range(height - 1, 0, -1):
        if np.asarray(nrrd_frame[row, :]).sum() > 0:
            row2 = row
            break

    for col in range(0, width):
        if np.asarray(nrrd_frame[:, col]).sum() > 0:
            col1 = col
            break

    for col in range(width - 1, 0, -1):
        if np.asarray(nrrd_frame[:, col]).sum() > 0:
            col2 = col
            break

    nrrd_frame = nrrd_frame[row1:row2, col1:col2]
    nrrd_center = (refactored_nrrd_center[0] - col1, refactored_nrrd_center[1] - row1)

    return nrrd_frame, nrrd_center

def pre_calibrate_single_frame(img_frame, nrrd_frame, show=False):
    '''
    Transform the position of the brain in the nrrd frame to adapt image frame
    :param img_frame:
    :param nrrd_frame:
    :return:
    '''
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    threshold = get_adaptive_threshold(img_frame)
    ret, th = cv2.threshold(img_frame, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    th = cv2.erode(th, kernel, iterations=2)
    _, contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    img_2_show = img_frame.copy()

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 2e5 and cv2.contourArea(contours[i]) < 1e7:
            x_t, y_t, w_t, h_t = cv2.boundingRect(contours[i])
            point_t = (int(x_t + 0.5 * w_t), int(y_t + 0.5 * h_t))
            if is_in_center(point_t, img_2_show):
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.rectangle(img_2_show, (x, y), (x + w, y + h), (255, 255, 0), 5)
                if show:
                    show_img(img_2_show, False)
    cur_h = 0
    cur_w = 0
    (height, length) = nrrd_frame.shape
    for row in range(height):
        if np.asarray(nrrd_frame[row, :]).sum() > 0:
            cur_h += 1
    for col in range(length):
        if np.asarray(nrrd_frame[:, col]).sum() > 0:
            cur_w += 1
    try:
        w_factor = float(w)/float(cur_w)
        h_factor = float(h) / float(cur_h)
    except:
        img_2_show = img_frame.copy()

        for i in range(len(contours)):
            # print(cv2.contourArea(contours[i]))
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(img_2_show, (x, y), (x + w, y + h), (255, 255, 0), 5)
            show_img(img_2_show, False)

    refactored_nrrd_center = (int(NRRD_CERTER_POSITION[2] * w_factor), int(NRRD_CERTER_POSITION[1] * h_factor))

    nrrd_frame = cv2.resize(nrrd_frame, (int(nrrd_frame.shape[1] * w_factor), int(nrrd_frame.shape[0] * h_factor)))

    nrrd_frame, nrrd_center = get_pure_brain_nrrd(nrrd_frame, refactored_nrrd_center)

    nrrd_frame_clean = nrrd_frame.copy()

    cv2.circle(nrrd_frame, nrrd_center, 10, (255, 255, 255), 10, 1)


    canvas = np.zeros((img_frame.shape[0], img_frame.shape[1])).astype(np.uint8)
    canvas_clean = canvas.copy()
    nrrd_size = nrrd_frame.shape
    canvas[0:nrrd_size[0], 0:nrrd_size[1]] = nrrd_frame
    canvas_clean[0:nrrd_size[0], 0:nrrd_size[1]] = nrrd_frame_clean

    canvas_center = (int(canvas.shape[1] * 0.5), int(canvas.shape[0] * 0.5))

    shift_col =  canvas_center[0] - nrrd_center[0]
    shift_row= canvas_center[1] - nrrd_center[1]
    M = np.float32([[1, 0, shift_col], [0, 1, shift_row]])
    canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))
    canvas_clean = cv2.warpAffine(canvas_clean, M, (canvas.shape[1], canvas.shape[0]))
    return canvas, canvas_clean

def affin_transform(img_frame, nrrd_frame):
    '''
    transform the shape of nrrd_frame to fit the tissue image frame
    :param img_frame:
    :param nrrd_frame:
    :return:
    '''
    hull = get_convex_hull(img_frame, False, False)
    point_img_list = calculate_intersection_point(hull, img_frame, False)
    x = int(img_frame.shape[1] * 0.5)
    y = int(img_frame.shape[0] * 0.5)
    point_img_list.append((x, y))
    point_img_list = np.float32(point_img_list)

    nrrd_frame_reference, nrrd_frame_clean = pre_calibrate_single_frame(img_frame, nrrd_frame, False)

    hull_nrrd = get_convex_hull(nrrd_frame_reference, False, False, 1)

    point_nrrd_list = calculate_intersection_point(hull_nrrd, nrrd_frame_reference, False)
    x = int(img_frame.shape[1] * 0.5)
    y = int(img_frame.shape[0] * 0.5)
    point_nrrd_list.append((x, y))
    point_nrrd_list = np.float32(point_nrrd_list)

    tform = PiecewiseAffineTransform()
    tform.estimate(point_img_list, point_nrrd_list)
    out = warp(nrrd_frame_clean, tform, output_shape=nrrd_frame_reference.shape)

    return out

def calculate_shift(img_dir):
    '''
    Calculate the index shift between image and nrrd
    :param img_dir:
    :return:
    '''
    def z_key(elem):
        return -float(elem.split(',')[-1].strip().split('.tif')[0])
    tif_list = os.listdir(img_dir)
    tif_list.sort(key=z_key)
    i = 0
    for tif_name in tif_list:
        last_temp = tif_name.split(',')[-1].strip().split('.tif')[0]
        if float(last_temp) == 0:
            index = i
            break
        i += 1
    shift = NRRD_CERTER_POSITION[0] - index
    return shift

def get_transformed_nrrd_list(img_dir, save_dir):
    '''
    get a list of transformed nrrd file
    :param img_dir:
    :return:
    '''
    if len(os.listdir(save_dir)) > 0:
        transformed_nrrd_list = []
        def sort_key(elem):
            return int(elem.split('.tif')[0])

        img_dir_list = os.listdir(save_dir)
        img_dir_list.sort(key=sort_key )
        for img_dir in img_dir_list:
            img = cv2.imread(os.path.join(save_dir, img_dir))
            img = np.asarray(img).astype(np.float32)
            transformed_nrrd = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.
            transformed_nrrd_list.append(transformed_nrrd)
    else:

        def z_key(elem):
            return -float(elem.split(',')[-1].strip().split('.tif')[0])
        tif_list = os.listdir(img_dir)
        tif_list.sort(key=z_key)

        f = open("nrrd_reference/nrrd_plot.pickle", 'rb')
        nrrd_plot_data = pk.load(f)
        shift = calculate_shift(img_dir)

        transformed_nrrd_list = []
        print("Calculating........")
        for i in tqdm(range(len(tif_list))):
            tif_name = tif_list[i]
            nrrd_frame = nrrd_plot_data[i + shift]
            img_frame = cv2.imread(os.path.join(img_dir, tif_name))
            transformed_nrrd = affin_transform(img_frame, nrrd_frame)
            transformed_nrrd_list.append(transformed_nrrd)
        print("Finished!")

    return transformed_nrrd_list

def save_tif(img_dir, save_folder="transformed_nrrd"):
    '''
    save the transformed nrrd frames into save_folder
    :param img_dir:
    :param save_folder:
    :return:
    '''
    transformed_nrrd_list = get_transformed_nrrd_list(img_dir, save_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for nrrd_index in range(len(transformed_nrrd_list)):
        img = transformed_nrrd_list[nrrd_index]
        img = np.asarray(img * 255).astype(np.uint8)
        skimage.io.imsave(os.path.join(save_folder, '%d.tif'%nrrd_index), img, plugin='tifffile')

def present(img_dir, root_dir, save_dir, plot_bead=False):
    '''
    Use for presenting single brain images
    :param img_dir: The folder holding the brain images
    :param root_dir: actually the root dir here means [root dir] + [folder of brain id]
    :param save_dir: The folder used for saving result given by save_main function. The present function will detect whether there are existing availabe result.
    :param plot_bead: if true, the result given by bead_finder will be presented on the image
    :return:
    '''
    def z_key(elem):
        return -float(elem.split(',')[-1].strip().split('.tif')[0])
    tif_list = os.listdir(img_dir)
    tif_list.sort(key=z_key)

    weight = 0.5

    transformed_nrrd_list = get_transformed_nrrd_list(img_dir, save_dir)
    index = 0
    if plot_bead:
        layer_dict = get_pixel_points_layer_dict(root_dir)

    while(True):

        tif_name = tif_list[index]
        img_frame = cv2.imread(os.path.join(img_dir, tif_name))

        transformed_nrrd = transformed_nrrd_list[index]

        img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY) / 255.

        img_frame_sum = cv2.addWeighted(img_gray.copy().astype(np.float32), weight, transformed_nrrd * 6., 1 - weight, 0)

        if plot_bead:
            tif_key = -z_key(tif_name)
            if tif_key in layer_dict:
                for point in layer_dict[tif_key]:
                    cv2.circle(img_frame_sum, point, 15, (255, 255, 255), thickness=5)

        key = show_img(img_frame_sum, True)
        if key == 101:
            weight -= 0.01
            if weight < 0:
                weight = 1.
        elif key == 113:
            weight += 0.01
            if weight > 1:
                weight = 0.
        elif key == 115:
            break
        elif key == 97:
            index += 1
            if index >= len(tif_list):
                index = 0
        elif key == 100:
            index -= 1
            if index < 0:
                index = len(tif_list) - 1

def save_main(root_dir="/home/silasi/brain_imgs/processed_images/"):
    '''
    The function for calculating and saving the nrrd image.
    The structure for holding data should be:
    [root dir] / [id for the brain] / [id for the brain]-aligned[id for the image].tif
    :param root_dir:
    :return:
    '''
    save_folder = "/" + root_dir.split('/')[0]
    for item in root_dir.split('/')[1: -2]:
        save_folder = os.path.join(save_folder, item)
    save_folder = os.path.join(save_folder, "transformed_nrrd")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    name_list = os.listdir(root_dir)
    for name in name_list:
        img_dir = os.path.join(root_dir, name, "3 - Processed Images", "7 - Counted Reoriented Stacks Renamed")
        save_dir = os.path.join(save_folder, name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print("working on " + name + "...")
        save_tif(img_dir, save_dir)

def present_main(root_dir="/home/silasi/brain_imgs/processed_images/"):
    '''
    The function for presenting the transformed nrrd and make it a mask on the real tissue image
    :param root_dir:
    :return:
    '''
    save_folder = "/" + root_dir.split('/')[0]
    for item in root_dir.split('/')[1: -2]:
        save_folder = os.path.join(save_folder, item)
    save_folder = os.path.join(save_folder, "transformed_nrrd")
    name_list = os.listdir(root_dir)
    for name in name_list:
        print(name)
        img_dir = os.path.join(root_dir, name, "3 - Processed Images", "7 - Counted Reoriented Stacks Renamed")
        save_dir = os.path.join(save_folder, name)
        present(img_dir, os.path.join(root_dir, name), save_dir)


if __name__ == '__main__':
    present_main()
    # save_main()