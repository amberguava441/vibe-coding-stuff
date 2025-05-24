import cv2 as cv
from dcb import *
import os
import shutil
import time

def cameraIni(cn=1):
    cap = cv.VideoCapture(cn)

    if not cap.isOpened():
        return 1, None
    else:
        ret, _ = cap.read()
        if not ret:
            return 2, None

        return 0, cap
    
def cameraRel(cap):
    if cap:
        cap.release()
        return 0
    return 1

def getFrame(cap):
    ret, frame = cap.read()
    if ret:
        return 0, frame
    return 1, None

def createEnv(cnl):
    cl = []

    cameradir = os.path.abspath("camera")
    os.makedirs(cameradir, exist_ok=True)
    shutil.rmtree(os.path.join(cameradir, 'test'), ignore_errors=True)
    for cn in cnl:
        os.makedirs(os.path.join(cameradir, str(cn)), exist_ok=True)
        os.makedirs(os.path.join(cameradir, 'test', str(cn)), exist_ok=True)
        os.makedirs(os.path.join(cameradir, 'calibration', str(cn)), exist_ok=True)
        if cn< 100:
            flag, c = cameraIni(cn)
        elif cn<200:
            flag, c = deepcameraIni(cn-100)
        cl.append([cn, flag, c, None, None, None, None])

    return cl, cameradir

def destroyEnv(cl):
    for cn, _, c, _, _, _, _ in cl:
        if cn<100:
            cameraRel(c)
        elif cn<200:
            deepcameraRel(c)

def testEnv():
    cnl = []
    cnl.append(0)
    cnl.append(1)
    cl, _ = createEnv(cnl)

    for cn, cflag, c, _, _, _, _ in cl:
        if 0==cflag:
            fflag, frame = getFrame(c)
            if 0==fflag:
                cv.imwrite(f"camera/test/{cn}/frame-test.jpg", frame)
        print(f"cn-{cn}|cflag-{0==cflag}|fflag-{0==fflag}")

    destroyEnv(cl)

def testCamera(cn, count=0, rate=120):
    flag, c = cameraIni(cn)

    if count==0:
        count=rate

    if 0==flag:
        for k in range(0,count):
            flag, frame = getFrame(c)
            if flag:
                break
            cv.imwrite(f"camera-{cn}-frame-{k}.jpg", frame)
            time.sleep(1/rate)

    cameraRel(c)


if __name__ == "__main__":
    testCamera(0)
    # testEnv()
