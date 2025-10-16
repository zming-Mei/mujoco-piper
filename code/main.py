import mujoco
import mujoco.viewer
import time
import ikpy.chain
import numpy as np

# Global variables
model = None
data = None
move_step = 0.05
target_pos_desired = None  # Cache target position
ik_update_flag = False  # Flag to trigger IK recalculation

def key_callback(key: int):
    global model, data, move_step, target_pos_desired, ik_update_flag
    if data is None or model is None:
        return

    try:
        target_jnt_id = model.joint('target_joint').id
        qpos_start = model.jnt_qposadr[target_jnt_id]
        current_pos = data.qpos[qpos_start:qpos_start + 3].copy()
    except Exception as e:
        print(f"Error accessing target joint: {e}")
        return

    # Update position based on key
    moved = False
    if key in (87, 119):      # W/w → +Y
        current_pos[1] += move_step
        moved = True
    elif key in (83, 115):    # S/s → -Y
        current_pos[1] -= move_step
        moved = True
    elif key in (65, 97):     # A/a → -X
        current_pos[0] -= move_step
        moved = True
    elif key in (68, 100):    # D/d → +X
        current_pos[0] += move_step
        moved = True
    elif key in (81, 113):    # Q/q → +Z
        current_pos[2] += move_step
        moved = True
    elif key in (69, 101):    # E/e → -Z
        current_pos[2] -= move_step
        moved = True

    if moved:
        # Write back position
        data.qpos[qpos_start:qpos_start + 3] = current_pos
        data.qpos[qpos_start + 3:qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]
        
        # Update cached target and flag for IK recalculation
        target_pos_desired = current_pos.copy()
        ik_update_flag = True
        print(f"Target moved to: {current_pos}")


def main():
    global model, data, target_pos_desired, ik_update_flag

    # --- Paths ---
    model_path = '/home/zming/mujoco/piper/piper_description/mujoco_model/scene.xml'
    urdf_path = '/home/zming/mujoco/piper/piper_description/urdf/piper_no_gripper_description.urdf'

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if model.nu != 6:
        raise ValueError(f"Expected 6 actuators, but model has {model.nu}.")

    # Load IK chain
    my_chain = ikpy.chain.Chain.from_urdf_file(
        urdf_path,
        active_links_mask=[False] + [True] * 6
    )

    # Initialize
    data.ctrl[:] = np.zeros(model.nu)
    mujoco.mj_forward(model, data)
    
    # Initialize target position
    target_pos_desired = data.body("target").xpos.copy()
    desired_ctrl = data.qpos[:6].copy()  # Initial joint angles

    # Control parameters
    kp = 2.0
    dt = model.opt.timestep if model.opt.timestep > 0 else 1.0 / 60.0
    
    # Rendering control
    physics_steps_per_render = 4  # Render every N physics steps
    step_counter = 0
    
    # Timing
    last_time = time.time()
    frame_times = []

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Only recalculate IK when target moves
            if ik_update_flag:
                try:
                    target_angles = my_chain.inverse_kinematics(
                        target_position=target_pos_desired
                    )
                    desired_ctrl = target_angles[1:7]
                    ik_update_flag = False
                except Exception as e:
                    print(f"IK failed: {e}")
            
            # Smooth control
            ctrl_error = np.array(desired_ctrl) - data.ctrl
            data.ctrl[:] += kp * ctrl_error * dt
            
            # Physics step
            mujoco.mj_step(model, data)
            step_counter += 1
            
            # Render at lower frequency
            if step_counter >= physics_steps_per_render:
                viewer.sync()
                step_counter = 0
                
                # Monitor frame time
                current_time = time.time()
                frame_time = current_time - last_time
                frame_times.append(frame_time)
                last_time = current_time
                
                # Print FPS every 100 frames
                if len(frame_times) >= 100:
                    avg_frame_time = np.mean(frame_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    print(f"Average FPS: {fps:.1f}")
                    frame_times = []
            
            # Maintain real-time factor
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()