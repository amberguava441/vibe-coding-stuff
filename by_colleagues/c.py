from cb import *
from calibration import *
from scipy.io import savemat, loadmat
from ct import *

from inference import mask_to_lane_edges, test_model
from segment_anything import SamPredictor
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

class LaneDetectionTester:
    def __init__(self):
        """Initialize the lane detection tester with RealSense camera and SlimSAM model."""
        
        # Initialize SlimSAM model
        print("Loading SlimSAM model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_slimsam_model()
        self.predictor = SamPredictor(self.model)
        
        # # Create output directories
        # self.output_dir = "lane_detection_results"
        # self.images_dir = os.path.join(self.output_dir, "images")
        # self.data_dir = os.path.join(self.output_dir, "data")
        
        # os.makedirs(self.images_dir, exist_ok=True)
        # os.makedirs(self.data_dir, exist_ok=True)
        
        # Default point prompt (center-bottom of typical image)
        # NOTE: Click point
        self.input_point = np.array([[320, 300]])  # Adjust based on your image size
        self.input_label = np.array([1])
        
        print(f"Using device: {self.device}")
        #print(f"Output directory: {self.output_dir}")
        
    def load_slimsam_model(self):
        """Load the SlimSAM model."""
        try:
            model_path = "./checkpoints/SlimSAM-77.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = torch.load(model_path, map_location=self.device)
            
            # Handle module wrapper if present
            if hasattr(model.image_encoder, 'module'):
                model.image_encoder = model.image_encoder.module
            
            model.to(self.device)
            model.eval()
            
            # Apply custom forward method (from inference.py)
            import types
            
            def forward(self, x):
                x = self.patch_embed(x)
                if self.pos_embed is not None:
                    x = x + self.pos_embed
                for blk in self.blocks:
                    x, qkv_emb, mid_emb, x_emb = blk(x)
                x = self.neck(x.permute(0, 3, 1, 2))
                return x
            
            model.image_encoder.forward = types.MethodType(forward, model.image_encoder)
            
            return model
            
        except Exception as e:
            print(f"Error loading SlimSAM model: {e}")
            print("Make sure the model file exists at 'checkpoints/SlimSAM-77.pth'")
            raise
    
    # def detect_lanes(self, rgb_image):
    #     """
    #     Detect lanes in the given RGB image.
        
    #     Args:
    #         rgb_image (numpy.ndarray): RGB image from camera
            
    #     Returns:
    #         tuple: (left_points, right_points, visualization_image)
    #     """
    #     try:
    #         # Set image for the predictor
    #         self.predictor.set_image(rgb_image)
            
    #         # Perform segmentation with point prompt
    #         with torch.no_grad():
    #             masks, scores, logits = self.predictor.predict(
    #                 point_coords=self.input_point,
    #                 point_labels=self.input_label,
    #                 box=None,
    #                 multimask_output=False,
    #             )
            
    #         # Extract the mask
    #         mask = masks.squeeze()
            
    #         # Convert mask to lane edge points
    #         # TODO: Modify
    #         left_points, right_points = mask_to_lane_edges(
    #             mask
    #         )
            
    #         points = left_points
    #         for point in right_points:
    #             points.append(point)
             
    #         return points
            
    #     except Exception as e:
    #         print(f"Error in lane detection: {e}")
    #         return [], [], None

    def detect_lanes(self, rgb_image):
        """
        Detect lanes in the given RGB image.
        
        Args:
            rgb_image (numpy.ndarray): RGB image from camera
            
        Returns:
            list: Combined list of lane points [(x,y), ...] or empty list []
        """
        try:
            # Set image for the predictor
            self.predictor.set_image(rgb_image)
            
            # Perform segmentation with point prompt
            with torch.no_grad():
                masks, scores, logits = self.predictor.predict(
                    point_coords=self.input_point,
                    point_labels=self.input_label,
                    box=None,
                    multimask_output=False,
                )
            
            # Extract the mask
            mask = masks.squeeze()
            
            # Convert mask to lane edge points
            left_points, right_points = mask_to_lane_edges(mask)
            
            # Ensure we always have valid lists
            if left_points is None:
                left_points = []
            if right_points is None:
                right_points = []
                
            # Combine all points
            points = list(left_points)  # Create a copy of left_points
            points.extend(right_points)  # Add all right_points
            
            return points
            
        except Exception as e:
            print(f"Error in lane detection: {e}")
            return []  # Return empty list on error

class camera():
    def __init__(self, cnl):
        self.cl, self.dir = createEnv(cnl)
        self.setCalibration()
        self.work = 0==self.check()

        self.cntok = {c[0]: k for k, c in enumerate(self.cl)}
        
    def setCalibration(self):
        for k, C in enumerate(self.cl):
            cn = C[0]
            dir = os.path.join(self.dir, 'calibration', f'{cn}.mat')
            if os.path.exists(dir):
                data = loadmat(dir)
                self.cl[k][3] = data['mtx']
                self.cl[k][4] = data['dist']
                self.cl[k][5] = data['rvecs']
                self.cl[k][6] = data['tvecs']
            else:
                cbpl = getcbpf(cn)
                if not cbpl:
                    continue
                else:
                    data = self.calibration(cn, cbpl)
                    self.cl[k][3] = data['mtx']
                    self.cl[k][4] = data['dist']
                    self.cl[k][5] = data['rvecs']
                    self.cl[k][6] = data['tvecs']

    def calibrationAll(self):
        for cn, _, c, _, _, _, _ in self.cl:
            self.calibrationOne(cn, c)

    def calibrationOne(self, cn, c):
        cbpl = getcbpf(cn)
        if not cbpl:
            cbpl = getcbp(cn, c)
        self.calibration(cn, cbpl)

    def calibration(self, cn, cbpl):
        ret, mtx, dist, rvecs, tvecs, size = calibration(cbpl)

        data = {
            'ret': ret,                             # 标定误差（重投影误差）
            'mtx': mtx,                             # 相机内参矩阵 3x3
            'dist': dist,                           # 畸变系数 [k1, k2, p1, p2, k3, ...]
            'rvecs': np.array(rvecs).squeeze(),     # 旋转向量
            'tvecs': np.array(tvecs).squeeze(),     # 平移向量
            'size': size                            # 图像尺寸 (width, height)
        }

        if ret is not None:
            savemat(os.path.join(self.dir, 'calibration', f'{cn}.mat'), data)
        return data

    def check(self):
        flag = 0
        for cn, cflag, c, mtx, _, _, _ in self.cl:
            if 0==cflag:
                if cn < 100:
                    fflag, _ = getFrame(c)
                    if 0==fflag and mtx is not None:
                        pass
                    else:
                        flag += 1
                elif cn < 200:
                    fflag, _, _ = getdeepFrame(c)
                    if 0==fflag and mtx is not None:
                        pass
                    else:
                        flag += 1
            else:
                fflag = 1
            
            print(f"cn-{cn}|cflag-{0==cflag}|fflag-{0==fflag}|calibration-{mtx is not None}")

        return flag

    def destroy(self):
        destroyEnv(self.cl)

    def getFrame(self, cn):
        cn, _, c, mtx, dist, _, _ = self.cl[self.cntok[cn]]
        if cn<100:
            flag, cframe = getFrame(c)
            dframe = None
        elif cn<200:
            flag, cframe, dframe = getdeepFrame(c)

        if mtx is not None and cframe is not None:
            cframe = cv.undistort(cframe, mtx, dist)
        if mtx is not None and dframe is not None:
            dframe = cv.undistort(dframe, mtx, dist)

        return flag, mtx is not None, cframe, dframe

# if __name__ == "__main__":
        
#     K = np.array([
#         [385.5,   0.0, 326.5],
#         [  0.0, 384.6, 242.5],
#         [  0.0,   0.0,   1.0]
#     ])
            
#     R = np.array([
#         [1, 0, 0],
#         [0, 0, -1],
#         [0, 1, 0]
#     ])

#     T = np.array([0, 0, 0])

#     cnl = []
#     cnl.append(100)
#     C = camera(cnl)

#     tester = LaneDetectionTester()

#     while True:
#         _,_, cframe, dframe = C.getFrame(100)
        
#         print(f"cframe shape: {cframe.shape}, dframe shape: {dframe.shape}")
        
#         lane = [(x, y, dframe[x,y]) for y, x in tester.detect_lanes(cframe)]
        
#         for x, y, _ in lane:
#             cframe[x, y] = [0, 0, 255]
#         cv.imwrite('a.png', cframe)
#         getF(lane, K, R, T)