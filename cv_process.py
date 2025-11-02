import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
import logging
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image
from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup
import open3d as o3d

# Suppress Ultralytics logging output
logging.getLogger("ultralytics").setLevel(logging.WARNING)
from PIL import Image


# -------------------- Model Initialization --------------------
def choose_model():
    """Initialize SAM predictor with proper parameters."""
    model_weight = 'ckpt/sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        model=model_weight,
        conf=0.1,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def set_classes(model, target_class):
    """Configure YOLO-World to detect a specific class."""
    model.set_classes([target_class])


def detect_objects(image_or_path, target_class=None):
    """
    Detect objects using YOLO-World.
    
    Args:
        image_or_path: Either a file path (str) or a NumPy array (BGR image).
        target_class: Optional string specifying the target object class.
    
    Returns:
        - List of detected bounding boxes in xyxy format with confidence and class.
        - Visualization image with bounding boxes drawn.
    """
    model = YOLO("ckpt/yolov8s-world.pt")
    if target_class:
        set_classes(model, target_class)
    
    results = model.predict(image_or_path)
    boxes = results[0].boxes
    vis_img = results[0].plot()  # Get detection visualization

    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.25:  # Confidence threshold
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": results[0].names[box.cls.item()]
            })

    return valid_boxes, vis_img


def process_sam_results(results):
    """
    Process SAM segmentation results to extract binary mask and centroid.
    
    Returns:
        - (cx, cy): Centroid coordinates of the segmented object.
        - mask: Binary mask as a NumPy uint8 array (0 or 255).
    """
    if not results or not results[0].masks:
        return None, None

    # Use the first mask (assuming single-object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contours and compute centroid
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def segment_image(image_path, output_mask='mask1.png'):
    """
    Segment a target object from an image using YOLO-World + SAM.
    
    Workflow:
        1. Prompt user for target class name.
        2. Run YOLO-World detection.
        3. If detections exist, select the highest-confidence one; otherwise, let user click.
        4. Use SAM to generate precise mask.
        5. Save and return the mask.

    """
    # Prompt user for target class
    target_class = input("\n===============\nEnter class name: ").strip()

    # Run YOLO detection
    detections, vis_img = detect_objects(image_path, target_class)
    cv2.imwrite('detection_visualization.jpg', vis_img)

    # Prepare RGB image for SAM
    if isinstance(image_path, str):
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    # Initialize SAM
    predictor = choose_model()
    predictor.set_image(image_rgb)

    # Choose detection method
    if detections:
        # Auto-select highest-confidence detection
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # Manual selection via mouse click
        cv2.imshow('Select Object', vis_img)
        point = []
        clicked = False

        def click_handler(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                clicked = True

        cv2.setMouseCallback('Select Object', click_handler)
        while not clicked:
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                raise ValueError("User cancelled selection")
        cv2.destroyAllWindows()

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # Save mask
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    else:
        print("[WARNING] Could not generate mask")

    return mask


def get_and_process_data(color_path, depth_path, mask_path, DEPTH_INTR, DEPTH_FACTOR):
    """
    Generate a sampled point cloud from RGB, depth, and mask inputs for grasp prediction.
    
    Supports both file paths and NumPy arrays as input.
    """
    NUM_POINT = 400

    # Load color image
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        # Assume input is BGR (OpenCV format); convert to RGB and normalize
        color = color_path.astype(np.float32)[..., [2, 1, 0]] / 255.0
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

    # Load mask
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path must be a string path or NumPy array!")

    # Validate dimensions
    print("\n=== Dimension Check ===")
    print("Depth shape:", depth.shape[::-1])
    print("Color shape:", color.shape[:2][::-1])
    print("Mask shape:", workspace_mask.shape[::-1])
    print("Expected camera resolution:", (DEPTH_INTR['width'], DEPTH_INTR['height']))

    # Create camera info and generate point cloud
    camera = CameraInfo(
        width=DEPTH_INTR['width'],
        height=DEPTH_INTR['height'],
        fx=DEPTH_INTR['fx'],
        fy=DEPTH_INTR['fy'],
        cx=DEPTH_INTR['ppx'],
        cy=DEPTH_INTR['ppy'],
        scale=DEPTH_FACTOR
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # Apply mask and depth validity
    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # Sample or upsample to fixed number of points
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]

    # Create Open3D point cloud for visualization
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # Prepare tensor for neural network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)

    end_points = {'point_clouds': cloud_sampled}
    return end_points, cloud_o3d


def collision_detection(gg, cloud_points, COLLISION_THRESH=0.02, VOXEL_SIZE=0.01):

    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]


if __name__ == '__main__':
    seg_mask = segment_image('color_img_path.jpg')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)