"""this class contains methods used to handle cartesian coordinates




    teh world coordinate system is defined using a right handed coordinate system wiht the x achese pointing away from the open



"""
import torch
from data_collection.utils.cartesian_constants import *


def get_xyz_from_matrix(matrix: torch.Tensor):
    pos = matrix[:-1, -1]
    return pos

def robot_frame_get_xyz(bot):
    matrix=torch.tensor(bot.arm.get_ee_pose())
    return get_xyz_from_matrix(matrix)

def world_frame_get_xyz (bot, right:bool):
    r_matrix = torch.tensor(bot.arm.get_ee_pose())
    t_matrix = torch.Tensor()
    if right:
        t_matrix = torch.tensor(RIGHT_ROBOT_TRANSFORMATION_MATRIX).double()
    else: 
        print("left")
        t_matrix = torch.tensor(LEFT_ROBOT_TRANSFORMATION_MATRIX).double()

    w_matrix = t_matrix @ r_matrix 

    return get_xyz_from_matrix(w_matrix)