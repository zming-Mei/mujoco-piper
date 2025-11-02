import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import cv2
import os
from datetime import datetime
from manipulator_grasp.env.piper_GS_env import PiperGraspEnv
import json

def setup_camera_d435(env):
    # Set camera intrinsics to match RealSense D435 in 640x480 mode
    camera_id = env.mj_model.camera('d435_rgb').id
    
    # D435 vertical FOV is 42 degrees; MuJoCo uses vertical FOV in degrees
    vertical_fov_deg = 42.0
    env.mj_model.cam_fovy[camera_id] = vertical_fov_deg
    
    print(f"Camera configured: FOV={vertical_fov_deg:.1f}°")


def capture_and_save_image(env, position_name, save_dir="captured_images"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Render image
    imgs = env.render()
    color_img = imgs['img']  # RGB
    color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    # Get camera pose in world frame
    camera_id = env.mj_model.camera('d435_rgb').id
    cam_pos_world = env.mj_data.cam_xpos[camera_id].copy()
    cam_rot_mat_world = env.mj_data.cam_xmat[camera_id].copy().reshape(3, 3)

    # Get piper base pose in world frame
    base_body_name = "piper"
    base_pos_world = env.mj_data.body(base_body_name).xpos.copy()
    base_quat_world = env.mj_data.body(base_body_name).xquat.copy()
    
    # MuJoCo xquat is [w, x, y, z]; convert to [x, y, z, w] for scipy
    base_quat_xyzw = np.array([base_quat_world[1], base_quat_world[2],
                               base_quat_world[3], base_quat_world[0]])
    base_rot_mat_world = R.from_quat(base_quat_xyzw).as_matrix()

    # Transform camera pose to base frame
    cam_pos_rel = base_rot_mat_world.T @ (cam_pos_world - base_pos_world)
    cam_rot_mat_rel = base_rot_mat_world.T @ cam_rot_mat_world
    cam_quat_rel = R.from_matrix(cam_rot_mat_rel).as_quat()  # [x, y, z, w]

    # Save RGB image
    color_path = os.path.join(save_dir, f"{position_name}_{timestamp}_rgb.png")
    cv2.imwrite(color_path, color_img_bgr)

    # Save relative pose
    pose_dict = {
        "camera_to_base": {
            "position": cam_pos_rel.tolist(),
            "quaternion_xyzw": cam_quat_rel.tolist(),
            "rotation_matrix": cam_rot_mat_rel.tolist()
        },
        "camera_in_world": {
            "position": cam_pos_world.tolist(),
            "rotation_matrix": cam_rot_mat_world.tolist()
        },
        "timestamp": timestamp,
        "position_name": position_name
    }

    pose_path = os.path.join(save_dir, f"{position_name}_{timestamp}_pose.json")
    with open(pose_path, 'w') as f:
        json.dump(pose_dict, f, indent=4)

    print(f"✓ Saved: {position_name}")
    print(f"  - RGB image: {color_path}")
    print(f"  - Camera-to-base pose: {pose_path}")

    # Display image (optional)
    cv2.imshow('Camera View - RGB Only', color_img_bgr)
    cv2.waitKey(1500)

    return color_img_bgr, pose_dict


def demo_multi_position_capture(env):
    print("=== Multi-position image capture demo ===")
    print("Configuring camera to D435 parameters...")
    
    setup_camera_d435(env)
    
    env.reset()
    print("\nWaiting for environment stabilization...")
    for _ in range(100):
        env.step()
    time.sleep(1)
    
    positions = [
        {
            'name': 'pos2_right',
            'coords': (0.3, 0.4, 0.4, np.pi*0.3, np.pi*0.9, 0),
            'description': 'Right-side view'
        }
    ]
    
    print(f"\nCapturing images at {len(positions)} positions...\n")
    
    all_images = []
    
    for i, pos in enumerate(positions, 1):
        print(f"\n[{i}/{len(positions)}] Moving to: {pos['description']}")
        print(f"  Target coordinates: x={pos['coords'][0]:.2f}, y={pos['coords'][1]:.2f}, "
              f"z={pos['coords'][2]:.2f}")
        
        env.move_to_position(*pos['coords'])
        
        print("  Waiting for arm stabilization...")
        for _ in range(50):
            env.step()
        time.sleep(0.5)
        
        print("  Capturing image...")
        color, pose = capture_and_save_image(env, pos['name'])
        all_images.append({'name': pos['name'], 'color': color})
        time.sleep(1)
    
    print("\n" + "="*60)
    print(f"✓ Capture completed! Total: {len(positions)} image sets")
    print("="*60)
    
    print("\nGenerating overview image...")
    create_overview_image(all_images, "captured_images/overview.png")
    
    return all_images


def create_overview_image(images, save_path):
    if not images:
        return
    
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    h, w = images[0]['color'].shape[:2]
    
    overview = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    for idx, img_data in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        img_copy = img_data['color'].copy()
        cv2.putText(img_copy, img_data['name'], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        overview[y_start:y_end, x_start:x_end] = img_copy
    
    cv2.imwrite(save_path, overview)
    print(f"✓ Overview saved: {save_path}")
    
    scale = min(1.0, 1200 / overview.shape[1])
    display_img = cv2.resize(overview, None, fx=scale, fy=scale)
    cv2.imshow('Overview - All Positions', display_img)
    cv2.waitKey(3000)


def main():
    print("="*60)
    print("Piper Arm Multi-Position Image Capture System")
    print("Camera: RealSense D435 (simulated)")
    print("="*60)
    
    env = PiperGraspEnv()
    
    try:
        images = demo_multi_position_capture(env)
        
        print("\nProgram will continue running, showing real-time view from last position...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        while True:
            env.step()
            
            if frame_count % 10 == 0:
                imgs = env.render()
                color_img = cv2.cvtColor(imgs['img'], cv2.COLOR_RGB2BGR)
                cv2.imshow('Real-time Camera View', color_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                capture_and_save_image(env, f"manual_{timestamp}")
            
            frame_count += 1
            time.sleep(0.002)
            
    except KeyboardInterrupt:
        print("\n\nInterrupt received, shutting down...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("✓ Environment closed")


if __name__ == '__main__':
    main()