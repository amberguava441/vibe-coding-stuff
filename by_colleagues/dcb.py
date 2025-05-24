import cv2 as cv
import pyrealsense2 as rs
import numpy as np

def deepcameraIni(dcn, R=640, C=480):
    dcl = rs.context().query_devices()
    if len(dcl) < dcn+1:
        return 1, None

    p = rs.pipeline()

    c = rs.config()
    c.enable_device(dcl[dcn].get_info(rs.camera_info.serial_number))
    c.enable_stream(rs.stream.depth, R, C, rs.format.z16, 30)
    c.enable_stream(rs.stream.color, R, C, rs.format.bgr8, 30)

    p.start(c)

    return 0, p

def getdeepFrame(p):
    frame = p.wait_for_frames()

    dframe = frame.get_depth_frame()
    cframe = frame.get_color_frame()

    if not dframe or not cframe:
        return 1, None, None

    dframe = dframe.get_data()
    cframe = cframe.get_data()

    dframe = np.array(dframe)*0.001
    cframe = np.array(cframe)

    return 0, cframe, dframe

def deepcameraRel(p):        
    p.stop()