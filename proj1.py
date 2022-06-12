from queue import Empty
from typing import final
import cv2
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join


def display_lines(image, lines):
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    # make sure array isn't empty
    group = []
    deviation = 0.15
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            slope = round(slope, 1)
            if slope <= deviation and slope >= -deviation:
                continue
            index = 0
            added = False
            group.sort(key=lambda y: y[0])

            for m, coord in group:
                if slope <= m + deviation and slope >= m - deviation:
                    _x1, _y1, _x2, _y2 = coord
                    points = [int((x1+_x1)/2), int((y1+_y1)/2),
                              int((x2+_x2)/2), int((y2+_y2)/2)]
                    x1_, y1_, x2_, y2_ = points
                    s = (y2_ - y1_) / (x2_ - x1_)
                    group[index] = (round(s, 1), points)
                    added = True
                index += 1
                if added:
                    break
            if not added:
                group.append((slope, [x1, y1, x2, y2]))

    for line in group:
        # print(line[0], line[1])
        x1, y1, x2, y2 = line[1]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    lines_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    cv2.imshow("Final", lines_image)
    print(len(group))
    return len(group)


def detect_stopsign(frame, imgContour):
    contours, h = cv2.findContours(
        frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            # Compute perimeter of contour and perform contour approximation
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Octagon
            if len(approx) == 8:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.circle(imgContour, (x+int(w/2), y + int(h/2)),
                           int(w/2), (255, 0, 0), 5)


def auto_canny(image, sigma):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def region_of_interst(img, vertices):
    mask = np.zeros_like(img)
    # channel_count=img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    mask_image = cv2.bitwise_and(img, mask)
    return mask_image


def convert_HSV_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    '''Red'''
    # Range for lower red
    red_lower = np.array([0, 50, 70])
    red_upper = np.array([9, 255, 255])
    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)
    # Range for upper range
    red_lower = np.array([159, 50, 70])
    red_upper = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, red_lower, red_upper)
    mask_red = mask_red1 + mask_red2
    return mask_red


def convert_HSV(frame, gray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    '''Yellow'''
    yellow_lower = np.array([10, 80, 80])
    yellow_upper = np.array([30, 255, 255])
    '''Red'''
    # Range for lower red
    red_lower = np.array([0, 50, 70])
    red_upper = np.array([9, 255, 255])
    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)
    # Range for upper range
    red_lower = np.array([159, 50, 70])
    red_upper = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, red_lower, red_upper)

    mask_white = cv2.inRange(gray, 190, 255)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red = mask_red1 + mask_red2

    mask_final = cv2.bitwise_or(mask_white, mask_yellow, mask_red)
    return mask_final


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def detect_lane(frame):
    result = 0
    # resize image
    resize = ResizeWithAspectRatio(frame, width=640)
    image = np.asarray(resize)
    frame1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Image processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernal_size = 7
    mask = convert_HSV(image, gray)
    blur = cv2.GaussianBlur(mask, (kernal_size, kernal_size), 0)
    ret, thresh = cv2.threshold(blur, 58, 255, cv2.THRESH_BINARY)
    edge = auto_canny(thresh, 0.33)

    # Stop sign
    mask2 = convert_HSV_red(image)
    blur2 = cv2.GaussianBlur(mask2, (kernal_size, kernal_size), 0)
    edge2 = auto_canny(blur2, 0.33)
    detect_stopsign(edge2, frame1)

    rows, cols = image.shape[:2]
    bottom_left = [cols*0, rows*1]
    top_left = [cols*0, rows*.5]
    bottom_right = [cols*1, rows*1]
    top_right = [cols*1, rows*.5]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    cropped = region_of_interst(edge, np.array(
        [vertices], np.int32))

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 70
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 150  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(cropped, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    result = display_lines(frame1, lines)
    return result

###########################################################################


def runon_image(path):
    frame = cv2.imread(path)
    # Change the codes here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections_in_frame = detect_lane(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Change the codes above
    #cv2.imshow("one image", frame)
    cv2.waitKey(0)
    return detections_in_frame


def runon_folder(path):
    files = None
    if(path[-1] != "/"):
        path = path + "/"
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    all_detections = 0
    for f in files:
        print(f)
        f_detections = runon_image(f)
        all_detections += f_detections
    return all_detections


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', help="requires path")
    args = parser.parse_args()
    folder = args.folder
    if folder is None:
        print("Folder path must be given \n Example: python proj1.py -folder images")
        sys.exit()

    if folder is not None:
        all_detections = runon_folder(folder)
        print("total of ", all_detections, " detections")

    cv2.destroyAllWindows()
