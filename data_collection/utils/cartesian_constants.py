import math


#rotation by 90 degree around z and translation along the new y achese by -0.5
RIGHT_ROBOT_TRANSFORMATION_MATRIX = [[math.cos(90), -math.sin(90), 0,0],
                                     [math.sin(90), math.cos(90), 0,-0.45],
                                     [0, 0,1, 0],
                                     [0, 0,0,1]]

#rotation by -90 degree around z and translation along the new y achese by 0.5
LEFT_ROBOT_TRANSFORMATION_MATRIX = [[math.cos(-90), -math.sin(-90),0, 0],
                                     [math.sin(-90), math.cos(-90), 0,0.45                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ],
                                     [0, 0,1, 0],
                                     [0, 0,0,1]]
