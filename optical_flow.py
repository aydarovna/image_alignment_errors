import cv2
import numpy as np


IMAGE_PATH = "./ImagePairs/8/8_input.png"
IMAGE_PATH_TARGET = "./ImagePairs/8/8_target.png"


def lucas_kanade_method(target, input):
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
 
    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
 
    # Create random colors
    color = np.random.randint(0, 255, (100, 3))
 
    # Take first frame and find corners in it
    old_frame = target
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
 
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    # Read new frame
    frame = input
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
 
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 10)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
 
    # Display the demo
    img = cv2.add(frame, mask)
    cv2.imshow("frame", img)
    cv2.waitKey(0)


def dense_optical_flow(method, target, input, params=[], to_gray=False):
    # hsv with const val
    hsv = np.zeros_like(target)
    hsv[..., 1] = 255
 
    # Preprocessing for exact method
    if to_gray:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = method(target, input, None, *params)
    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Use Hue and Value to encode the Optical Flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
 
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    v_values = np.sum(hsv[:, :, 2])
    area = input.shape[0] * input.shape[1]
    avg_brightness = v_values/area
    
    return bgr, avg_brightness
