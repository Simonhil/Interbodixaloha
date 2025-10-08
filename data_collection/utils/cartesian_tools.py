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


def world_frame_get_xyz (bot, right:bool):
    r_matrix = torch.tensor(bot.arm.get_ee_pose())
    t_matrix = torch.Tensor()
    if right:
        t_matrix = torch.tensor(RIGHT_ROBOT_TRANSFORMATION_MATRIX).double()
    else: 
        t_matrix = torch.tensor(LEFT_ROBOT_TRANSFORMATION_MATRIX).double()

    w_matrix = t_matrix @ r_matrix 

    return get_xyz_from_matrix(w_matrix)

def check_box_collision(bot, right:bool):
    curent_pos = world_frame_get_xyz(bot, right)
    abs_cur_pos = torch.abs(curent_pos)
    diff = torch.tensor(ABS_BORDER_VECTOR) - abs_cur_pos
    print(diff)
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

def make_frame_from_axis(axis_dir, origin):
    """
    axis_dir: 3-vector (direction for local z)
    origin: 3-vector (position of origin)
    returns: R (3x3), p (3,)
    """
    z = unit(axis_dir)
    # choose a reference vector not parallel to z
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x = unit(x)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))  # columns are x,y,z
    return R, np.asarray(origin, dtype=float)

def point_on_prismatic_axis_from_Scol(S_col):
    """
    For prismatic S = [0; v], choose a point p on axis by projecting world origin
    onto the line along direction v passing through origin (so p = 0).
    If you prefer another point, change this function.
    """
    # here we simply return the origin (0,0,0) as a reference point on the axis.
    return np.zeros(3)

def joint_frames_from_Slist_p_home(Slist, p_home):
    """
    Slist: (6, n) numpy array
    p_home: list of length n of either None or 3-array-like (points on axis in space frame)
    Returns:
      Rs: list of 3x3 rotation matrices
      Ps: list of 3-vectors (origins)
      Ts: list of 4x4 homogeneous transforms
    """
    Slist = np.asarray(Slist)
    n = Slist.shape[1]
    Rs = []
    Ps = []
    Ts = []
    for k in range(n):
        S = Slist[:, k]
        w = S[:3]
        v = S[3:]
        if norm(w) > 1e-12:  # revolute
            axis_dir = w
            if p_home[k] is None:
                # compute a point on axis from (w, v): q = (w x v)/||w||^2
                q = np.cross(w, v) / (norm(w)**2)
                origin = q
            else:
                origin = np.asarray(p_home[k], dtype=float)
        else:  # prismatic
            # axis direction = v (direction of translation)
            axis_dir = v
            if p_home[k] is None:
                origin = point_on_prismatic_axis_from_Scol(S)
            else:
                origin = np.asarray(p_home[k], dtype=float)

        R, p = make_frame_from_axis(axis_dir, origin)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        Rs.append(R)
        Ps.append(p)
        Ts.append(T)
    return Rs, Ps, Ts


def get_arm_poses(bot, right):
    joint_states = bot.arm.get_joint_positions()
    arm_joints = []
    p_home = [None]*6
    for k in range(6):
        S = robot_des.Slist[:,k]
        w = S[:3]; v = S[3:]
        if norm(w) > 1e-12:  # revolute
            q = np.cross(w, v) / (norm(w)**2)
            p_home[k] = q

        _,_,ts = joint_frames_from_Slist_p_home(robot_des.Slist,p_home )

    for i in range(6):
        if right:
            t_matrix = torch.tensor(RIGHT_ROBOT_TRANSFORMATION_MATRIX).double()
        else:
             t_matrix = torch.tensor(LEFT_ROBOT_TRANSFORMATION_MATRIX).double()
        world_pos = t_matrix @ torch.tensor(mr.FKinSpace(robot_des.M, robot_des.Slist, joint_states)).double()
        
        
        arm_joints.append(get_xyz_from_matrix(torch.tensor(mr.FKinSpace(robot_des.M, robot_des.Slist, joint_states[:i])).double()))
    return mr.FKinSpace(robot_des.M, robot_des.Slist, joint_states)