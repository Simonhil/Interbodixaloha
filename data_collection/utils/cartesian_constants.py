import math

import torch

#distances in Meter
#rotation by 90 degree around z and translation along the new y achese by -0.45 adn -0.3 in z 
RIGHT_ROBOT_TRANSFORMATION_MATRIX = torch.tensor([[math.cos(math.radians(90)), -math.sin(math.radians(90)), 0,0],
                                     [math.sin(math.radians(90)), math.cos(math.radians(90)), 0,-0.45],
                                     [0, 0,1, -0.3],
                                     [0, 0,0,1]]).double()

#rotation by -90 degree around z and translation along the new y achese by 0.45 and -0.3 in z
LEFT_ROBOT_TRANSFORMATION_MATRIX = torch.tensor([[math.cos(math.radians(-90)), -math.sin(math.radians(-90)),0, 0],
                                     [math.sin(math.radians(-90)), math.cos(math.radians(-90)), 0,0.45                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ],
                                     [0, 0,1, -0.3],
                                     [0, 0,0,1]]).double()

#distance border in y orrientation
X_BORDER_DISTANCE = 0.31
Y_BORDER_DISTANCE = 0.33
Z_BORDER_DISTANCE= 0.29

ABS_BORDER_VECTOR = [X_BORDER_DISTANCE, Y_BORDER_DISTANCE, Z_BORDER_DISTANCE]