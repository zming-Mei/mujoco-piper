### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
git switch -c main
cd mujoco-piper
conda env create --file environment.yml
conda activate piper
```

### Project Structure Overview

#### URDF and Main XML Files

- **Robot Arm Model**:  
  `manipulator_grasp/assets/agilex_piper/piper.xml`

- **Scene Configuration**:  
  `manipulator_grasp/assets/scenes/scene_grasp.xml`

#### Core Source Code

- **Robot Inverse Kinematics (IK) & Environment Logic**:  
  `manipulator_grasp/env/piper_grasp_env.py`

- **Image Processing Utilities**:  
  `cv_process.py`

- **End-to-End Grasping Pipeline with GraspNet**:  
  `main_piper_graspnet.py`





### To run the grasping pipeline:

```bash
python main_piper_graspnet.py
```


1. **Input Prompt**  
   The program will prompt you to enter a **class name** (e.g., "cup", "bottle", "box").

2. **Object Recognition**  
   - ✅ **If recognition succeeds**:  
     GraspNet automatically executes the **grasp-and-place** sequence.
   - ❌ **If recognition fails**:  
     You will be asked to **manually select an object** by **clicking on it** in the displayed image window.

3. **Output Visualization Files**  
   - **Detection result**: `detection_visualization.jpg`  
     (Shows bounding boxes and recognized class labels)
   - **Segmentation mask**: `mask1.png`  
     (Binary or instance segmentation mask of the selected object)


