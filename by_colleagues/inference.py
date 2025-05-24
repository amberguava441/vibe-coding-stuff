import numpy as np
import cv2
import torch
import torch.nn as nn
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
# from segment_anything_kd.modeling.image_encoder import add_decomposed_rel_pos
import matplotlib.pyplot as plt
import torch_pruning as tp
from skimage.measure import label
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) for two binary masks.

    Parameters:
        mask1 (numpy.ndarray): The first binary mask.
        mask2 (numpy.ndarray): The second binary mask.

    Returns:
        float: The IoU score.
    """
    # Make sure the input masks have the same shape
    assert mask1.shape == mask2.shape, "Both masks must have the same shape."

    # Calculate the intersection and union of the masks
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Compute the IoU score
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
            
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def plot_points(points, H, output_filename: str = 'scatter_plot.png'):
    """
    Plot a list of (x, y) points and save the figure as an image.
    
    Args:
        points: List of tuples, where each tuple contains (x, y) coordinates
        output_filename: Name of the output image file
    """
    # Extract x and y coordinates from the list of tuples
    x = [point[0] for point in points]
    y = [H - point[1] for point in points]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the scatter plot
    scatter = ax.scatter(x, y, c='blue', marker='o', s=100, alpha=0.7)
    
    # Customize the plot
    ax.set_title('Scatter Plot of (X,Y) Points', fontsize=14)
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Auto-adjust axis limits with a small margin
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    margin = 0.1  # 10% margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    
    # Save the figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{output_filename}'")

# def mask_to_lane_edges(mask):
#     """
#     将二值车道掩码图像转换为左右车道边缘点在相机坐标系下的二维坐标。
    
#     参数:
#         mask (numpy.ndarray): 二值掩码图 (H×W), 其中车道区域为1，背景为0。
#         K (numpy.ndarray): 相机内参矩阵 (3×3), 包含 [fx,  0, cx; 0, fy, cy; 0, 0, 1]。
#         R (numpy.ndarray): 相机外参旋转矩阵 (3×3), 世界坐标系到相机坐标系的旋转。
#         t (numpy.ndarray): 相机外参平移向量 (3×1), 世界坐标系原点在相机坐标系中的位置。
#         C (numpy.ndarray): 世界坐标系下的光心点坐标 (3×1)
    
#     返回:
#         tuple: (left_edge_points, right_edge_points)
#                left_edge_points 和 right_edge_points 为列表，包含若干 [x, y] 坐标点（单位：米）。
#                第一个列表为左车道边缘点集合，第二个列表为右车道边缘点集合。
#     """
#     # 从内参矩阵提取相机参数
#     H, W = mask.shape[:2]
    
#     # 1. 提取最大连通域作为车道区域
#     mask_uint8 = mask.astype('uint8')
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
#     if num_labels <= 1:
#         # 没有检测到有效连通域
#         return ([], [])
#     # 找到除背景以外面积最大的连通域标签 (背景标签为0)
#     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     lane_mask = (labels == largest_label).astype('uint8')
    
#     # 可选：进行形态学操作，平滑边缘去除毛刺（根据需要）
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     # lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    
#     # 2. 扫描每一行，获取左右边缘像素坐标
#     H, W = lane_mask.shape
#     left_pixels = []
#     right_pixels = []
#     for v in range(H):
#         if v < 60:
#             continue
#         else:
#             row = lane_mask[v]
#             # 获取该行中属于车道区域的像素列索引
#             indices = np.where(row > 0)[0]
#             if indices.size == 0:
#                 continue                # 该行没有车道区域
#             left_u = indices[0]         # 最左侧像素列
#             right_u = indices[-1]       # 最右侧像素列
#             left_pixels.append((left_u, v))
#             right_pixels.append((right_u, v))
    
#     # # TODO: 进行滤波
#     # points = left_pixels
#     # for point in right_pixels:
#     #     points.append(point)

#     # for point in points:
#     #     print(point)   

#     # plot_points(points, 500)
    
#     return (left_pixels, right_pixels)

def mask_to_lane_edges(mask):
    """
    将二值车道掩码图像转换为左右车道边缘点在相机坐标系下的二维坐标。
    
    参数:
        mask (numpy.ndarray): 二值掩码图 (H×W), 其中车道区域为1，背景为0。
    
    返回:
        tuple: (left_edge_points, right_edge_points)
               left_edge_points 和 right_edge_points 为列表，包含若干 [x, y] 坐标点（单位：像素）。
               第一个列表为左车道边缘点集合，第二个列表为右车道边缘点集合。
               如果没有检测到车道，返回 ([], [])
    """
    # Initialize empty lists to ensure we always return valid lists
    left_pixels = []
    right_pixels = []
    
    try:
        # 从内参矩阵提取相机参数
        H, W = mask.shape[:2]
        
        # 1. 提取最大连通域作为车道区域
        mask_uint8 = mask.astype('uint8')
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels <= 1:
            # 没有检测到有效连通域
            return (left_pixels, right_pixels)  # Return empty lists
        
        # 找到除背景以外面积最大的连通域标签 (背景标签为0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        lane_mask = (labels == largest_label).astype('uint8')
        
        # 可选：进行形态学操作，平滑边缘去除毛刺（根据需要）
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. 扫描每一行，获取左右边缘像素坐标
        H, W = lane_mask.shape
        
        for v in range(H):
            if v < 60:
                continue
            else:
                row = lane_mask[v]
                # 获取该行中属于车道区域的像素列索引
                indices = np.where(row > 0)[0]
                if indices.size == 0:
                    continue                # 该行没有车道区域
                
                left_u = indices[0]         # 最左侧像素列
                right_u = indices[-1]       # 最右侧像素列
                
                # Only add points if they are valid
                if 0 <= left_u < W and 0 <= v < H:
                    left_pixels.append((left_u, v))
                if 0 <= right_u < W and 0 <= v < H and right_u != left_u:
                    right_pixels.append((right_u, v))
        
        # Always return two lists (even if empty)
        return (left_pixels, right_pixels)
        
    except Exception as e:
        print(f"Error in mask_to_lane_edges: {e}")
        # Return empty lists on any error
        return ([], [])



def mask_to_bev_cloud(mask, K, cam_h=1.5, stride=3):
    """
    把掩码中所有⽐特>0 的像素投影到 (x,y) 平面。
    - mask : 2D uint8 (0/255)
    - K    : 相机内参
    - cam_h: 相机离地⾼度 (m)
    - stride: 像素抽样步长，1 表全部像素，=3 表每 3×3 取⼀个像素
    返回: N×2 numpy array [[x,y]...]
    """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    ys, xs = np.where(mask > 0)
    # 仅保留地平线以下 & stride 抽样
    idx = (ys > cy) & ((ys % stride) == 0) & ((xs % stride) == 0)
    u, v = xs[idx], ys[idx]
    if u.size == 0:
        return np.zeros((0, 2))
    Y = cam_h * fy / (v - cy)
    X = (u - cx) * Y / fx
    pts = np.column_stack([X, Y])
    return pts.astype(float)

def plot_lane_boundaries_bev(left_pts, right_pts, save_path=None, forward_range=15):
    """
    仅显示摄像头前 forward_range 米 (默认 15 m) 内的左右车道边界
    x 向右，y 向前
    """
    left_pts  = np.asarray(left_pts)
    right_pts = np.asarray(right_pts)

    # ---------- 过滤超过 forward_range m 的点 ----------
    if left_pts.size:
        left_pts = left_pts[left_pts[:, 1] <= forward_range + 1e-6]
    if right_pts.size:
        right_pts = right_pts[right_pts[:, 1] <= forward_range + 1e-6]

    plt.figure(figsize=(6, 10))
    if left_pts.size:
        plt.plot(left_pts[:, 0],  left_pts[:, 1],  'r-', label='Left Lane')
    if right_pts.size:
        plt.plot(right_pts[:, 0], right_pts[:, 1], 'g-', label='Right Lane')

    plt.xlabel("X (Right, m)")
    plt.ylabel("Y (Forward, m)")
    plt.title("Lane Boundaries in Camera Coordinates (Top View)")
    plt.grid(True)
    plt.axis('equal')

    # 先设定 y 轴范围，再翻转方向
    plt.ylim(0, forward_range)      # 0 m（车前保险杠处）到 forward_range m
    plt.gca().invert_yaxis()        # 让 “向前” 朝上

    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_bev_15m(left_pts=None, right_pts=None, cloud_pts=None,
                 save_path=None, forward_range=15):
    # ----------- 统一转 np.array & 取前两列 (x,y) -----------
    def _to_xy(pts):
        if pts is None or len(pts) == 0:
            return np.empty((0, 2))
        pts = np.asarray(pts)
        if pts.shape[1] == 3:        # (x,y,z) -> (x,y)
            pts = pts[:, :2]
        return pts

    left_xy  = _to_xy(left_pts)
    right_xy = _to_xy(right_pts)
    cloud_xy = _to_xy(cloud_pts)

    # ----------- 过滤纵深 > forward_range 的点 -----------
    mask_y = lambda arr: arr[:, 1] <= forward_range + 1e-6
    left_xy  = left_xy[mask_y(left_xy)]   if left_xy.size  else left_xy
    right_xy = right_xy[mask_y(right_xy)] if right_xy.size else right_xy
    cloud_xy = cloud_xy[mask_y(cloud_xy)] if cloud_xy.size else cloud_xy

    # ----------- 绘图 -----------
    plt.figure(figsize=(6, 10))
    if left_xy.size:
        plt.plot(left_xy[:, 0],  left_xy[:, 1],  'r-', label='Left Lane')
    if right_xy.size:
        plt.plot(right_xy[:, 0], right_xy[:, 1], 'g-', label='Right Lane')
    if cloud_xy.size:
        plt.scatter(cloud_xy[:, 0], cloud_xy[:, 1],
                    s=2, c='b', alpha=0.5, label='Point Cloud')

    plt.xlabel("X (Right, m)")
    plt.ylabel("Y (Forward, m)")
    plt.title("BEV within %.1f m of Camera" % forward_range)
    plt.grid(True)
    plt.axis('equal')

    # y 轴：0–forward_range，然后翻转方向(向前朝上)
    plt.ylim(0, forward_range)
    plt.gca().invert_yaxis()

    plt.legend(loc='upper right')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def test_model(input_point,path,K):

    device = torch.device("cuda")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    model_path = "checkpoints/SlimSAM-77.pth"
    SlimSAM_model = torch.load(model_path)
    SlimSAM_model.image_encoder = SlimSAM_model.image_encoder.module
    SlimSAM_model.to(device)
    SlimSAM_model.eval()
    print("model_path:",model_path)

    def forward(self, x):

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x,qkv_emb,mid_emb,x_emb = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        
        return x

    import types
    funcType = types.MethodType
    SlimSAM_model.image_encoder.forward = funcType(forward, SlimSAM_model.image_encoder)

    example_inputs = torch.randn(1, 3, 1024, 1024).to(device)
    ori_macs, ori_size = tp.utils.count_ops_and_params(SlimSAM_model.image_encoder, example_inputs)
    print("MACs(G):",ori_macs/1e9,"Params(M):",ori_size/1e6)

    #mask_generator = SamAutomaticMaskGenerator(teacher_model)
    mask_generator = SamAutomaticMaskGenerator(
    model=SlimSAM_model,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

    predictor = SamPredictor(SlimSAM_model)

    with torch.no_grad():
        print(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(image)

        
################ Point Prompt ################
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box = None,
        multimask_output=False,
    )
        #mask = masks.squeeze()
        #points = mask_to_lane_cloud(mask, K, camera_height=1.5, stride=3)
        #plot_lane_cloud_bev(points, save_path="images/lane_bev.png")
        # 1. SlimSAM 得到掩码
        mask1 = masks.squeeze()
        left_tup,right_tup = mask_to_lane_edges(mask1, K, camera_height=0.7)
        # 过滤纵向距离超过 20m 的点
        left_tup  = [pt for pt in left_tup if 0 <= pt[1] <= 20]
        right_tup = [pt for pt in right_tup if 0 <= pt[1] <= 20]
        #plot_lane_boundaries_bev(left_tup, right_tup, save_path="images/lane_boundaries.png")
        mask = masks.squeeze().astype(np.uint8) * 255
        # 整块车道点云
        lane_cloud = mask_to_bev_cloud(mask, K, cam_h=0.7, stride=3)

        plot_bev_15m(left_pts=left_tup,
             right_pts=right_tup,
             cloud_pts=lane_cloud,   
             forward_range=15)       # 如需别的范围可改

        print("left_tup:",left_tup)
        print("right_tup:",right_tup)
        
        print("masks.shape:",masks.shape)   
        plt.figure(figsize=(15,10))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("images/"+'demo_point' + ".png")


'''################ Box Prompt ################
        input_box = np.array([75, 275, 1725, 850])
        masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box = input_box,
        multimask_output=False,
    )   
        plt.figure(figsize=(15,10))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("images/"+'demo_box' + ".png")'''


'''################ Segment everything prompt ################
        masks = mask_generator.generate(image)
        plt.figure(figsize=(15,10))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show()
        plt.tight_layout()
        plt.savefig("images/"+"demo_everything" + ".png")'''


# K = np.array([
#     [1341.44246083, 0., 655.70222442],
#     [0., 1340.99460995, 859.40825835],
#     [0., 0., 1.]
# ])

# if __name__ == '__main__':
#     input_point = np.array([[750, 1000]])
#     input_picture_path = 'images/trail.jpg'
#     test_model(input_point, input_picture_path,K)