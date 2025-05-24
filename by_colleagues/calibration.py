
from cb import *
import numpy as np

def getcbp(cn, c, count=30, rate=1):
    cbpl = []

    for k in range(0,count):
        if cn < 100:
            flag, frame = getFrame(c)
        elif cn <200:
            flag, frame, _ = getdeepFrame(c)
        if flag:
            break
        cbpl.append(frame)
        cv.imwrite(f"camera/calibration/{cn}/frame-{k}.jpg", frame)
        print(k)
        time.sleep(1/rate)

    return cbpl

def getcbpf(cn):
    cbpl = []

    k = 0
    while os.path.exists(f"camera/calibration/{cn}/frame-{k}.jpg"):
        img = cv.imread(f"camera/calibration/{cn}/frame-{k}.jpg")
        if img is not None:
            cbpl.append(img)
        k += 1
    
    return cbpl

def calibration(cbpl, cbs=(11, 8),cbl = 0.003):
    p3D = []  # 世界坐标系
    p2D = []  # 图像坐标系

    cbp = np.zeros((cbs[0] * cbs[1], 3), np.float32)
    cbp[:, :2] = np.indices(cbs).T.reshape(-1, 2)*cbl

    for frame in cbpl:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, cbs, None)
        if ret:
            p2D.append(corners)
            p3D.append(cbp)

    try:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(p3D, p2D, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs, gray.shape[::-1]
    except:
        return None, None, None, None, None, None

    
