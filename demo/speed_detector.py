
import cv2 as cv
import numpy as np
import os

WIDTH = 160       
HEIGHT = 120        

uploads_folder = '/Users/riyasachdeva/Desktop/edge_project/demo/uploads'
for file_name in os.listdir(uploads_folder):
    if file_name.endswith('.mp4'):
        INPUT_PATH = os.path.join(uploads_folder, file_name)
        break
else:
    raise FileNotFoundError("No .mp4 file found in the uploads folder")

CAM = cv.VideoCapture(INPUT_PATH)
UBASE_WIDTH = 60    
LBASE_WIDTH = 320   
UOFFSET = 45        
LOFFSET = 20        
MAX_ACC = 0.2      

SRC_UL = [(WIDTH - UBASE_WIDTH) / 2, UOFFSET]
SRC_LL = [(WIDTH - LBASE_WIDTH) / 2, HEIGHT - LOFFSET]
SRC_UR = [(WIDTH + UBASE_WIDTH) / 2, UOFFSET]
SRC_LR = [(WIDTH + LBASE_WIDTH) / 2, HEIGHT - LOFFSET]

DST_UL = [0, 0]
DST_LL = [0, HEIGHT]
DST_UR = [WIDTH, 0]
DST_LR = [WIDTH, HEIGHT]

VELOCITY_CUTOFF_PCT = 67

def make_velocity_detector():

    pts1 = np.float32([SRC_UL, SRC_LL, SRC_UR, SRC_LR])
    pts2 = np.float32([DST_UL, DST_LL, DST_UR, DST_LR])

    M = cv.getPerspectiveTransform(pts1, pts2)

    prev = None
    v_last = 0.0

    def detect_velocity(image):
        nonlocal prev, v_last
        curr_bgr = cv.warpPerspective(image, M, (160, 120))
        curr = cv.cvtColor(curr_bgr, cv.COLOR_BGR2GRAY)

        if prev is None:
            prev = curr
            v_last = 0.0
            return v_last, curr_bgr, np.zeros_like(image)

        flow = cv.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.5, 0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        v = mag * np.sin(ang)

        ar = np.arange(-20.0, 20.0, 0.50, dtype=np.float64)
        his = np.histogram(v, bins=ar)

        for i, n in enumerate(his[0]):
            bgr = (255, 255, 0)
            if his[1][i] < 0:
                bgr = (0, 255, 255)

            cv.rectangle(   image, #curr_bgr,
                            (i*2, HEIGHT),
                            (i*2, HEIGHT - int(n / 10)),
                            bgr, #(0, 255, 255),
                            cv.FILLED)

        hsv = np.zeros_like(image)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(np.abs(v), None, 0, 255, cv.NORM_MINMAX)
        hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


        v_abs = np.absolute(v)
        v = v[v_abs >= np.percentile(v_abs, VELOCITY_CUTOFF_PCT)]

        v_max = v_last + MAX_ACC
        v_min = v_last - MAX_ACC
        v = np.clip(v, v_min, v_max)
        if v.size > 0:
            v_avg = v.mean()
        else:
            if v_last > 0:
                v_avg = max(v_last - MAX_ACC, 0)
            elif v_last < 0:
                v_avg = min(v_last + MAX_ACC, 0)
            else:
                v_avg = 0

        prev = curr
        v_last = v_avg
        return v_last, curr_bgr, hsv_bgr

    return detect_velocity

# def get_velocity():
get_velocity = make_velocity_detector()

while 1:

    ret, image = CAM.read()
    if not ret:
        break

    velocity, top_view, hsv_bgr = get_velocity(image)
    # print('velocity:', velocity)
    with open("v.txt", "a") as f:
        f.write(f"{velocity}\n")

    if velocity >= 0:
        cv.rectangle(   image,
                        (80, 0),
                        (int(velocity * 8) + 80, 4),
                        (0, 0, 255),
                        cv.FILLED)
    else:
        cv.rectangle(   image,
                        (80, 0),
                        (int(velocity * 8) + 80, 4),
                        (255, 0, 0),
                        cv.FILLED)

    vis = np.concatenate((image, top_view, hsv_bgr), axis=1)
    cv.imshow('Velocity Detection using Optical Flow', vis)

    k = cv.waitKey(30) & 0xff
    if k == 27: 
        break


CAM.release()
cv.destroyAllWindows()