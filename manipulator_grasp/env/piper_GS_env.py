import os.path
import sys
import time
import numpy as np
import casadi
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import pinocchio as pin
from pinocchio import casadi as cpin
from manipulator_grasp.arm.motion_planning import *
from pathlib import Path
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))


class IKSolver:
    """Simplified IK solver using Pinocchio + CasADi"""

    def __init__(self, urdf_path: str, ee_frame_name: str = 'grasp_link'):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        
        self._setup_optimizer()
        
    def _setup_optimizer(self):
        """Configure CasADi optimizer: define variables, cost function, joint limits, and solver options."""
        cq = casadi.SX.sym("q", self.model.nq, 1)
        cTf = casadi.SX.sym("tf", 4, 4)
        
        cpin.framesForwardKinematics(self.cmodel, self.cdata, cq)
        error_vec = cpin.log6(
            self.cdata.oMf[self.ee_frame_id].inverse() * cpin.SE3(cTf)
        ).vector
        
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        
        error_func = casadi.Function("error", [cq, cTf], [error_vec])
        position_cost = casadi.sumsqr(error_func(self.var_q, self.param_tf))
        regularization = casadi.sumsqr(self.var_q)
        
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit
        ))
        
        self.opti.minimize(1000 * position_cost + 0.01 * regularization)
        
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 10000,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)
    
    def _normalize_q(self, q):
        q = np.array(q).flatten()
        if q.size == 8:
            return q
        elif q.size == 7:
            return np.append(q, -q[-1])
        else:
            raise ValueError(f"Joint dimension must be 7 or 8, got {q.size}")
    
    def solve(self, q_init, target_pose: pin.SE3):
        """
        Solve inverse kinematics: given an initial joint configuration and a target end-effector pose,
        return the optimal joint angles satisfying joint limits.
        """
        q_init = self._normalize_q(q_init)
        self.opti.set_initial(self.var_q, q_init)
        self.opti.set_value(self.param_tf, casadi.DM(target_pose.homogeneous))
        
        try:
            sol = self.opti.solve_limited()
            q_sol = np.array(self.opti.value(self.var_q)).flatten()
        except RuntimeError as e:
            print(f"IK solve failed: {e}")
            q_sol = q_init
            
        return q_sol
    
    def forward_kinematics(self, q):
        """
        Compute forward kinematics: given joint angles, return the end-effector pose as a pinocchio.SE3 object.
        """
        q = self._normalize_q(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id]


class PiperGraspEnv:
    """Piper robotic arm grasping environment in MuJoCo"""

    def __init__(self):
        BASE = Path(__file__).parent
        mj_filename = (BASE / '../assets/scenes/scene_3DGS.xml').resolve()
        urdf_path = (BASE / '../assets/agilex_piper/piper_description/urdf/piper_description.urdf').resolve()
        mj_filename = str(mj_filename.resolve())
        urdf_path = str(urdf_path.resolve())
        
        self.ik_solver = IKSolver(urdf_path, ee_frame_name='grasp_link')

        self.mj_model = mujoco.MjModel.from_xml_path(mj_filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.camera_id = self.mj_model.camera('d435_rgb').id
        
        self.height, self.width = 512, 512
        self.mj_renderer = mujoco.renderer.Renderer(
            self.mj_model, height=self.height, width=self.width
        )
        self.mj_depth_renderer = mujoco.renderer.Renderer(
            self.mj_model, height=self.height, width=self.width
        )
        self.mj_depth_renderer.enable_depth_rendering()
        
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self._setup_camera()
        
        self.q_current = np.array([0, 1.128, -1.327, 0, 1.636, 0, 0])
    
    def _setup_camera(self):
        self.mj_viewer.cam.lookat[:] = [1.8, 1.1, 1.7]
        self.mj_viewer.cam.azimuth = 210
        self.mj_viewer.cam.elevation = -35
        self.mj_viewer.cam.distance = 1.2
        self.mj_viewer.sync()
    
    def reset(self):
        self.q_current = np.array([0, 1.128, -1.327, 0, 1.636, 0, 0])
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
    
    def step(self, action=None):
        if action is not None:
            self.q_current = action
            self.mj_data.ctrl = action
        
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_viewer.sync()
        
    def render(self):
        self.mj_renderer.update_scene(self.mj_data, self.camera_id)
        self.mj_depth_renderer.update_scene(self.mj_data, self.camera_id)
        
        rgb = self.mj_renderer.render()
        depth = self.mj_depth_renderer.render()
        
        rgb_rotated = rgb
        depth_rotated = depth
        return {
            'img': rgb_rotated,
            'depth': depth_rotated
        }
    
    def move_to_position(self, *args):
        """
        Move the robot to a target end-effector pose.
        Accepts either a 4x4 homogeneous transformation matrix or (x, y, z, rx, ry, rz) in radians.
        """
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            target_pose = args[0]
            translation = target_pose[:3, 3]
            rotation = target_pose[:3, :3]
        elif len(args) == 6:
            x, y, z, rx, ry, rz = args
            translation = np.array([x, y, z])
            rotation = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        else:
            raise ValueError("Invalid input: expected 4x4 matrix or (x,y,z,rx,ry,rz)")
        
        target_pose = pin.SE3(rotation, translation)
        q_new = self.ik_solver.solve(self.q_current, target_pose)
        self.q_current = q_new[:7]
        self.step(self.q_current)
    
    def pose_to_joint(self, *args):
        """
        Convert a desired end-effector pose to joint angles without executing motion.
        Returns a 7-dimensional joint configuration.
        """
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            target_pose = args[0]
            translation = target_pose[:3, 3]
            rotation = target_pose[:3, :3]
        elif len(args) == 6:
            x, y, z, rx, ry, rz = args
            translation = np.array([x, y, z])
            rotation = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        else:
            raise ValueError("Invalid input")
        
        target_pose = pin.SE3(rotation, translation)
        q_new = self.ik_solver.solve(self.q_current, target_pose)
        return q_new[:7]
    
    def get_joint_state(self):
        return self.mj_data.qpos[:7]
    
    def get_end_pose(self):
        """
        Get current end-effector pose by computing forward kinematics from MuJoCo joint states.
        """
        q_full = np.append(self.mj_data.qpos[:7], -self.mj_data.qpos[6])
        pose = self.ik_solver.forward_kinematics(q_full)
        print(f"End-effector position: {pose.translation}")
        print(f"End-effector orientation:\n{pose.rotation}")
        return pose
    
    def gripper_control(self, command):
        """
        Control the gripper: accepts either a float (target opening width in meters)
        or a boolean (True = open, False = close).
        """
        if isinstance(command, float):
            self.q_current[-1] = np.clip(command, 0, 0.037)
            self.step(self.q_current)
        elif isinstance(command, bool):
            target = 0.037 if command else 0.0
            step_size = 0.00003
            for _ in range(1000):
                if command and self.q_current[-1] >= target:
                    break
                if not command and self.q_current[-1] <= target:
                    break
                self.q_current[-1] += step_size if command else -step_size
                self.q_current[-1] = np.clip(self.q_current[-1], 0, 0.037)
                self.step(self.q_current)
        return self.q_current
    
    def trajectory_planner(self, q_init: list, q_goal: list, duration: float = 2.0):
        """
        Execute a smooth joint-space trajectory using quintic polynomial interpolation.
        """
        parameter = JointParameter(q_init, q_goal)
        velocity_parameter = QuinticVelocityParameter(duration)
        trajectory_parameter = TrajectoryParameter(parameter, velocity_parameter)
        planner = TrajectoryPlanner(trajectory_parameter)
        
        time_steps = int(duration / 0.002) + 1
        times = np.linspace(0.0, duration, time_steps)
        
        for t in times:
            q_interp = planner.interpolate(t)
            if isinstance(q_interp, np.ndarray):
                self.step(q_interp)
    
    def close(self):
        if self.mj_viewer:
            self.mj_viewer.close()
        if self.mj_renderer:
            self.mj_renderer.close()
        if self.mj_depth_renderer:
            self.mj_depth_renderer.close()


if __name__ == '__main__':
    env = PiperGraspEnv()
    env.reset()
    
    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        env.close()