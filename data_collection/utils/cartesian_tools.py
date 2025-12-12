"""this class contains methods used to handle cartesian coordinates




    teh world coordinate system is defined using a right handed coordinate system wiht the x achese pointing away from the open



"""
import numpy as np
import torch
from data_collection.utils.cartesian_constants import *
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
import modern_robotics as mr
from numpy.linalg import norm


def get_xyz_from_matrix(matrix: torch.Tensor):
    pos = matrix[:-1, -1]
    return pos

def robot_frame_get_xyz(bot):
    matrix=torch.tensor(bot.arm.get_ee_pose())
    return get_xyz_from_matrix(matrix)


def convert_to_world (r_matrix, right:bool):
    t_matrix = torch.Tensor()
    if right:
        t_matrix = torch.tensor(RIGHT_ROBOT_TRANSFORMATION_MATRIX).double()
    else: 
        t_matrix = torch.tensor(LEFT_ROBOT_TRANSFORMATION_MATRIX).double()

    w_matrix = t_matrix @ r_matrix 

    return get_xyz_from_matrix(w_matrix)

def world_frame_get_matrix(bot, right:bool):
    r_matrix = torch.tensor(bot.arm.get_ee_pose())
    if right:
        t_matrix = RIGHT_ROBOT_TRANSFORMATION_MATRIX
    else: 
        t_matrix = LEFT_ROBOT_TRANSFORMATION_MATRIX

    w_matrix = t_matrix @ r_matrix 

    return w_matrix


def world_frame_get_xyz (bot, right:bool):
    w_matrix = world_frame_get_matrix(bot, right)
    return get_xyz_from_matrix(w_matrix)

def check_box_collision(bot, right:bool):
    curent_pos = world_frame_get_xyz(bot, right)
    abs_cur_pos = torch.abs(curent_pos)
    diff = torch.tensor(ABS_BORDER_VECTOR) - abs_cur_pos
    has_negative = (diff < 0).any()

    return has_negative.item()



robot_des : mrd.ModernRoboticsDescription = getattr(mrd, 'vx300s')







def norm(v):
    return np.linalg.norm(v)

def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def cartesian_6D_to_rotmatrix(orientation):
    "given a vector containing the 6 values of the first two colums withing a rotation matrix this method converts them back to the coresponding matrix"
    # Normalize first vector
    b1 = torch.nn.functional.normalize(orientation[:3], dim=-1)
    
    # Make second vector orthogonal to first
    dot = (b1 * orientation[3:]).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(orientation[3:] - dot * b1, dim=-1)
    
    # Third vector via cross product
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack as rotation matrix
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # shape (..., 3, 3)
    return rot_mat

def rot_matrix_to_6D(rot_matrix):
   return rot_matrix[..., :3, :2].reshape(*rot_matrix.shape[:-2], 6)

def get_ee_6D_representation(bot, right:bool):
    #TODO adjust back to worls frame for final use
    w_matrix = torch.tensor(bot.arm.get_ee_pose())
    pos_vektor = w_matrix[:3, 3]
    rot_matrix = w_matrix[:3, :3]
    orientation = rot_matrix_to_6D(rot_matrix)
    return torch.tensor(pos_vektor), torch.tensor(orientation)


def get_ee_6D_total(bot, right:bool):
    pos, orientation = get_ee_6D_representation(bot, right)
    return torch.cat((pos, orientation))






def batch_6d_to_rot_matrix(six_d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D representation to 3x3 rotation matrix.
    Args:
        six_d: Tensor shape (..., 6)
    Returns:
        rot: Tensor shape (..., 3, 3)
    """
    a1 = six_d[..., :3]
    a2 = six_d[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, p=2, dim=-1)                    # (...,3)
    # remove component of a2 along b1
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1      # (...,3)
    b2 = torch.nn.functional.normalize(a2 - proj, p=2, dim=-1)             # (...,3)
    b3 = torch.cross(b1, b2, dim=-1)                     # (...,3)

    rot = torch.stack((b1, b2, b3), dim=-1)              # (...,3,3) columns are b1,b2,b3
    return rot






def batch_convert_6D_vector_to_Transformationmatrix(vectors):
    """
    Given a single 9D vector or a batch of 9D vectors (xyz + 6D), return homogeneous transform(s).
    Args:
        vectors: Tensor or list; shape (9,) or (N,9)
                 layout per row: [x,y,z, a1,a2,a3, b1,b2,b3]
    Returns:
        transforms: Tensor shape (4,4) or (N,4,4)
    """
    vec = torch.as_tensor(vectors)
    single = False
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
        single = True
    assert vec.shape[1] == 9,"Each vector must have length 9 (xyz + 6D). Got shape: " + str(vec.shape)
     

    pos = vec[:, :3]                 # (N,3)
    sixd = vec[:, 3:]                # (N,6)

    rot = batch_6d_to_rot_matrix(sixd) # (N,3,3)

    N = rot.shape[0]
    transforms = torch.eye(4, dtype=rot.dtype, device=rot.device).unsqueeze(0).repeat(N, 1, 1)  # (N,4,4)
    transforms[:, :3, :3] = rot
    transforms[:, :3, 3] = pos

    return transforms[0] if single else transforms

def convert_joint_to_ee_matrix(arm, joints, right=False):
    cartesian = []
    joints = np.array(joints)
    if right:
       joints = joints[:, 7:13]
    else:
        joints = joints[:, :6]
    for joint_state in joints:
        cartesian.append( mr.FKinSpace(arm.robot_des.M, arm.robot_des.Slist, joint_state))
    return torch.tensor(cartesian)

def batch_c_matrix_to_joint(bot, actions):
    joints = []
    for action in actions:
        joint_state, valid = bot.arm.set_ee_pose_matrix(action, execute=False, blocking = True,)
        if valid:
            joints.append(joint_state)
        else:
            raise "no valid position"
    return joints