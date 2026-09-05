# PiPER-X model source

The arm and gripper meshes, inertial values, joint transforms, and limits come
from AgileX Robotics' `agx_arm_urdf` repository:

- Repository: https://github.com/agilexrobotics/agx_arm_urdf
- Source revision: `f6642ce0d7872c686f29c99e9e10cd23d1d49313`
- Source files: `piper_x/urdf/piper_x_description.urdf` and
  `piper_x/urdf/piper_x_with_gripper_description.xacro`

`piper_x.urdf` flattens those files, rewrites ROS package paths to local paths,
removes Xacro mimic tags (RoboTwin drives them explicitly), and adds a fixed
SAPIEN wrist-camera frame. See `LICENSE.agx_arm_urdf` for the upstream license.
