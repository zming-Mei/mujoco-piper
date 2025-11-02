import os.path
import sys
import cv2
import time
import numpy as np
import spatialmath as sm
import torch
import copy
from graspnetAPI import GraspGroup
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from scipy.optimize import least_squares
from manipulator_grasp.env import piper_grasp_env
import pinocchio as pin
from PIL import Image

import spatialmath as sm
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from cv_process import segment_image

from manipulator_grasp.arm.motion_planning import *


# ================= Process input data and generate point cloud ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    Load RGB image, depth map, and mask (either as file paths or NumPy arrays),
    then generate a sampled point cloud and associated data for grasp prediction.
    """
    # Load color image
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32) / 255.0
    else:
        raise TypeError("color_path must be a string path or NumPy array!")

    # Load depth map
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path must be a string path or NumPy array!")

    # Load workspace mask
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path must be a string path or NumPy array!")

    # Camera intrinsic parameters (simulated camera with vertical FOV = π/4)
    height, width = color.shape[:2]
    fovy = np.pi / 4
    focal = height / (2.0 * np.tan(fovy / 2.0))
    c_x, c_y = width / 2.0, height / 2.0
    intrinsic = np.array([
        [focal, 0.0, c_x],
        [0.0, focal, c_y],
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0

    # Generate organized point cloud from depth
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # Apply workspace and depth validity mask
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # Uniformly sample or upsample to fixed number of points
    NUM_POINT = 3000
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # Create Open3D point cloud for visualization
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # Prepare tensor input for neural network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)

    end_points = {
        'point_clouds': cloud_sampled,
        'cloud_colors': color_sampled
    }

    return end_points, cloud_o3d


# ================= Generate grasp predictions ====================
def generate_grasps(end_points, cloud, visual=False):
    """
    Main grasp inference pipeline:
    1. Load pre-trained GraspNet model
    2. Run forward pass and decode grasp predictions
    3. Perform collision detection
    4. Apply NMS and sort by confidence score
    5. Filter grasps by approach angle (near-vertical only)
    """
    # Load model
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    checkpoint = torch.load('ckpt/checkpoint-rs.tar')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # Forward inference
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # Collision detection
    COLLISION_THRESH = 0.05
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        gg = gg[~collision_mask]

    # Non-maximum suppression and sorting
    gg = gg.nms().sort_by_score()

    # Filter grasps with near-vertical approach direction
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # upward Z-axis
    angle_threshold = np.deg2rad(30)  # 30 degrees
    filtered = []
    for grasp in all_grasps:
        approach_dir = grasp.rotation_matrix[:, 0]  # grasp x-axis = approach direction
        cos_angle = np.clip(np.dot(approach_dir, vertical), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)

    if not filtered:
        return None

    # Sort by score and select the best grasp
    filtered.sort(key=lambda g: g.score, reverse=True)
    best_grasp = filtered[0]

    # Create new GraspGroup with only the best grasp
    new_gg = GraspGroup()
    new_gg.add(best_grasp)

    if visual:
        grippers = [g.to_open3d_geometry() for g in filtered]
        o3d.visualization.draw_geometries([cloud, *grippers])

    return new_gg


# ================= Execute grasp in simulation ====================
def execute_grasp(env: piper_grasp_env.PiperGraspEnv, gg: GraspGroup):
    """
    Execute a full grasp sequence in simulation:
    - Move to pre-grasp pose
    - Approach object
    - Close gripper
    - Lift and relocate object
    - Release and reset
    """
    gripper_maxW = 0.038

    # Define coordinate frames using spatialmath
    # World-to-camera transform
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([1.15, 0.6, 1.6])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))

    # Camera-to-object transform (from grasp prediction)
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

    # World-to-robot-base transform
    t_wr = np.array([0.94, 0.6, 0.745])
    n_wr = np.array([1, 0, 0])
    o_wr = np.array([0, 1, 0])
    T_wr = sm.SE3.Trans(t_wr) * sm.SE3(sm.SO3.TwoVectors(x=n_wr, y=o_wr))
    T_wr_inv = T_wr.inv()

    # Compute object pose in robot base frame with additional rotations to align gripper
    T_bo = T_wr_inv * T_wc * T_co * sm.SE3.Rz(np.pi / 2) * sm.SE3.Rx(np.pi / 2) * sm.SE3.Rz(-np.pi / 2)

    # 1. Move to initial ready pose
    time1 = 2
    q0 = env.get_joint()
    q1 = np.array([0.0, 0.644, -0.62, 0.1, 0.57, 0, gripper_maxW])
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    time_array = [0.0, time1]
    planner_array = [planner1]
    total_time = sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        if timei == 0.0:
            continue
        for j in range(len(time_cumsum)):
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    env.step(planner_interpolate)
                break
    env.set_piper_qpos(q1)

    # 2. Move to pre-grasp pose (10 cm above grasp point)
    T2 = T_bo * sm.SE3(0.0, 0.0, -0.1)
    q2 = env.Pose_to_Joint_IK(T2.A)
    env.trajtory_planner(q_init=q1, q_goal=q2)

    # 3. Execute grasp (move to grasp pose and close gripper)
    T3 = T_bo
    q3 = env.Pose_to_Joint_IK(T3.A)
    env.trajtory_planner(q_init=q2, q_goal=q3)
    q3 = env.gripper_control(False)  # close gripper

    # 4. Lift object
    T4 = T_bo * sm.SE3(0.0, 0.0, -0.3)
    q4 = env.Pose_to_Joint_IK(T4.A)
    env.trajtory_planner(q_init=q3, q_goal=q4)

    # 5. Move horizontally to drop location
    T5 = sm.SE3.Trans(0.0, -0.4, T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    q5 = env.Pose_to_Joint_IK(T5.A)
    env.trajtory_planner(q_init=q4, q_goal=q5)

    # 6. Lower and release object
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5
    q6 = env.Pose_to_Joint_IK(T6.A)
    env.trajtory_planner(q_init=q5, q_goal=q6)
    q6 = env.gripper_control(True)  # open gripper

    # 7. Lift slightly and return to ready pose
    T7 = T5
    q7 = env.Pose_to_Joint_IK(T7.A)
    env.trajtory_planner(q_init=q6, q_goal=q7)
    env.trajtory_planner(q_init=q7, q_goal=q1)


if __name__ == '__main__':
    print(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
    env = piper_grasp_env.PiperGraspEnv()
    env.reset()
    for i in range(500):
        env.step()
    while True:
        # 1. Render RGB and depth images from simulator
        imgs = env.render()
        color_img = imgs['img']  # MuJoCo returns RGB
        depth_img = imgs['depth']
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)  # convert to BGR for OpenCV

        # 2. Segment target object using SAM
        mask_img = segment_image(color_img)

        # 3. Generate point cloud from RGB-D and mask
        end_points, cloud_o3d = get_and_process_data(color_img, depth_img, mask_img)

        # 4. Predict grasp pose
        gg = generate_grasps(end_points, cloud_o3d, visual=False)

        if gg is None:
            print('No valid grasp pose found.')
        else:
            # 5. Execute grasp in simulation
            execute_grasp(env, gg)
    env.close()