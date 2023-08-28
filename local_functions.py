import csv
import cv2
import os
import math

import numpy as np
import pandas as pd
from optical_flow import dense_optical_flow


MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.15
DIFF_THRESH = 3
BRIGHTNESS_THRESH = 2.0
ANGLE_THRESH = 0.001
ACCEPTABLE_EXP_DIFF_PERCENT = 5

directory = './ImagePairs' #local dir
folders = [f.path for f in os.scandir(directory) if f.is_dir()]
print(f'folders: {folders}')


def pre_processing(input, target):
    image_input = cv2.imread(input)
    image_target = cv2.imread(target)
    # convert to grayscale
    gray_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)

    return gray_input, gray_target

def draw_key_points(img, keypoint, color, thickness):
    for cur_key in keypoint:
        x = int(cur_key.pt[0])
        y = int(cur_key.pt[1])
        size = int(cur_key.size)
        cv2.circle(img, (x, y), size, color, thickness=thickness, lineType=8, shift=0)   
    return img  


def detect_alignment_error(image1, image2):
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(image1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2_gray, None)

    # dist_l2 = cv2.norm(descriptors1, descriptors2, cv2.NORM_HAMMING)
    # print(dist_l2, 'distl2')

    # # sanity check to visualize the keypoints
    # visual_img = draw_key_points(image1Gray, keypoints1, (255, 0, 0), 5)
    # cv2.imshow("Key Points", visual_img)
    # cv2.waitKey(0)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by hamming dist
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]
    # matched_image = cv2.drawMatches(image1, keypoints1, image2,
    #                                 keypoints2, matches, None, matchColor=(0, 255, 0),flags=2)
    # cv2.imshow("All Matches", matched_image)
    # cv2.waitKey(0)

    # Extract location of good matches and filter by diffy if rotation is small
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # initialize empty arrays for newpoints1 and newpoints2 and mask
    newpoints1 = np.empty(shape=[0, 2], dtype=np.float32)
    newpoints2 = np.empty(shape=[0, 2], dtype=np.float32)
    matches_mask = [0] * len(matches)

    for i in range(len(matches)):
        pt1 = points1[i]
        pt2 = points2[i]
        pt1x, pt1y = zip(*[pt1])
        pt2x, pt2y = zip(*[pt2])
        diff_y = np.float32(np.float32(pt2y) - np.float32(pt1y))
        if abs(diff_y) < DIFF_THRESH:
            newpoints1 = np.append(newpoints1, [pt1], axis=0).astype(np.uint8)
            newpoints2 = np.append(newpoints2, [pt2], axis=0).astype(np.uint8)
            matches_mask[i] = 1

    m, inliers = cv2.estimateAffinePartial2D(newpoints2, newpoints1)
    # warp img2 to match im1
    height, width, channels = image1.shape
    aligned_img = cv2.warpAffine(image2, m, (width, height))

    # Print angle
    row1_col0 = m[1, 0]
    angle = math.degrees(math.asin(row1_col0))

    if abs(angle) >= ANGLE_THRESH:
        error = True
    else:
        error = False

    return aligned_img, angle, error


def acceptable_exposure_error(input, target):

    def brightness_value(img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_values = np.sum(hsv_img[:, :, 2])
        area = img.shape[0] * img.shape[1]
        avg_brightness = v_values/area

        return avg_brightness

    exp1 = brightness_value(input)
    exp2 = brightness_value(target)
    if abs(exp1-exp2)/max(exp1, exp2)*100 <= ACCEPTABLE_EXP_DIFF_PERCENT:
        return True
    
    return False


def detect_motion(target, input):
    # bgr = farneback_optical_flow(img1, img2)
    method = cv2.optflow.calcOpticalFlowSparseToDense
    img, brightness = dense_optical_flow(method, target, input, to_gray=True)
    print(brightness)
    if brightness > BRIGHTNESS_THRESH:
        return True
    
    return False


list_correct = []
list_errors = []
data = pd.DataFrame(columns=['image_pairs', 'errors'])
for folder in folders:
    for filename in os.listdir(folder):
        if 'input' in filename:
            IMAGE_PATH_INPUT = os.path.join(folder, filename)
        elif 'target' in filename:
            IMAGE_PATH_TARGET = os.path.join(folder, filename)
    img1 = cv2.imread(IMAGE_PATH_TARGET)
    img2 = cv2.imread(IMAGE_PATH_INPUT)
    if not acceptable_exposure_error(img1, img2):
        list_errors.append(folder)
        df2 = pd.DataFrame([{'image_pairs': folder, 'errors': 'exposure'}])
        data = pd.concat([data, df2], ignore_index=True) 

    if detect_alignment_error(img1, img2)[2]:
        list_errors.append(folder)
        df3 = pd.DataFrame([{'image_pairs': folder, 'errors': 'alignment'}])
        data = pd.concat([data, df3], ignore_index=True)

    if detect_motion(img1, img2):
        list_errors.append(folder)
        df4 = pd.DataFrame([{'image_pairs': folder, 'errors': 'motion'}])
        data = pd.concat([data, df4], ignore_index=True)
    

print('data', data)
data.to_csv('errors.txt', index=False)  

    