from interbotix_xs_modules.xs_launch import (
    declare_interbotix_xsarm_robot_description_launch_arguments,
)
from interbotix_common_modules.launch import (
    AndCondition,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import (
  IfCondition,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


def launch_setup(context, *args, **kwargs):
    robot_model_leader_launch_arg = LaunchConfiguration('robot_model_leader')
    robot_model_follower_launch_arg = LaunchConfiguration('robot_model_follower')

    robot_name_leader_left_launch_arg = LaunchConfiguration('robot_name_leader_left')
    robot_name_leader_right_launch_arg = LaunchConfiguration('robot_name_leader_right')
    robot_name_follower_left_launch_arg = LaunchConfiguration('robot_name_follower_left')
    robot_name_follower_right_launch_arg = LaunchConfiguration('robot_name_follower_right')

    leader_modes_left_launch_arg = LaunchConfiguration('leader_modes_left')
    leader_modes_right_launch_arg = LaunchConfiguration('leader_modes_right')
    follower_modes_left_launch_arg = LaunchConfiguration('follower_modes_left')
    follower_modes_right_launch_arg = LaunchConfiguration('follower_modes_right')

    robot_description_leader_left_launch_arg = LaunchConfiguration(
        'robot_description_leader_left'
    )
    robot_description_leader_right_launch_arg = LaunchConfiguration(
        'robot_description_leader_right'
    )
    robot_description_follower_left_launch_arg = LaunchConfiguration(
        'robot_description_follower_left'
    )
    robot_description_follower_right_launch_arg = LaunchConfiguration(
        'robot_description_follower_right'
    )

    is_mobile = LaunchConfiguration('is_mobile').perform(context).lower() == 'true'

    xsarm_control_leader_left_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'launch',
                'xsarm_control.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_model': robot_model_leader_launch_arg,
            'robot_name': robot_name_leader_left_launch_arg,
            'mode_configs': leader_modes_left_launch_arg,
            'motor_configs': PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'config',
                f'{robot_model_leader_launch_arg.perform(context)}.yaml',
            ]),
            'use_rviz': 'false',
            'robot_description': robot_description_leader_left_launch_arg,
        }.items(),
        condition=IfCondition(LaunchConfiguration('launch_leaders')),
    )

    xsarm_control_leader_right_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'launch',
                'xsarm_control.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_model': robot_model_leader_launch_arg,
            'robot_name': robot_name_leader_right_launch_arg,
            'mode_configs': leader_modes_right_launch_arg,
            'motor_configs': PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'config',
                f'{robot_model_leader_launch_arg.perform(context)}.yaml',
            ]),
            'use_rviz': 'false',
            'robot_description': robot_description_leader_right_launch_arg,
        }.items(),
        condition=IfCondition(LaunchConfiguration('launch_leaders')),
    )

    xsarm_control_follower_left_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'launch',
                'xsarm_control.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_model': robot_model_follower_launch_arg,
            'robot_name': robot_name_follower_left_launch_arg,
            'mode_configs': follower_modes_left_launch_arg,
            'motor_configs': PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'config',
                f'{robot_model_follower_launch_arg.perform(context)}.yaml',
            ]),
            'use_rviz': 'false',
            'robot_description': robot_description_follower_left_launch_arg,
        }.items(),
    )

    xsarm_control_follower_right_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'launch',
                'xsarm_control.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_model': robot_model_follower_launch_arg,
            'robot_name': robot_name_follower_right_launch_arg,
            'mode_configs': follower_modes_right_launch_arg,
            'motor_configs': PathJoinSubstitution([
                FindPackageShare('interbotix_xsarm_control'),
                'config',
                f'{robot_model_follower_launch_arg.perform(context)}.yaml',
            ]),
            'use_rviz': 'false',
            'robot_description': robot_description_follower_right_launch_arg,
        }.items(),
    )

    leader_left_transform_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='leader_left_transform_broadcaster',
        arguments=[
            '-0.5',
            '0.25',
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            '1.0',
            '/world',
            ('/', LaunchConfiguration('robot_name_leader_left'), '/base_link'),
        ],
        output={'both': 'log'},
    )

    leader_right_transform_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='leader_right_transform_broadcaster',
        arguments=[
            '-0.5',
            '-0.25',
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            '1.0',
            '/world',
            ('/', LaunchConfiguration('robot_name_leader_right'), '/base_link'),
        ],
        output={'both': 'log'},
    )

    follower_left_transform_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='follower_left_transform_broadcaster',
        arguments=[
            '0.5',
            '0.25',
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            '1.0',
            '/world',
            ('/', LaunchConfiguration('robot_name_follower_left'), '/base_link'),
        ],
        output={'both': 'log'},
    )

    follower_right_transform_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='follower_right_transform_broadcaster',
        arguments=[
            '0.5',
            '-0.25',
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            '1.0',
            '/world',
            ('/', LaunchConfiguration('robot_name_follower_right'), '/base_link'),
        ],
        output={'both': 'log'},
    )

    # rs_actions = []
    # mobile_cams = [
    #     LaunchConfiguration('cam_high_name'),
    #     LaunchConfiguration('cam_left_wrist_name'),
    #     LaunchConfiguration('cam_right_wrist_name')
    # ]
    # all_cams = mobile_cams + [LaunchConfiguration('cam_low_name')]
    # camera_names = mobile_cams if is_mobile else all_cams
    # for camera_name in camera_names:
    #     rs_actions.append(
    #         Node(
    #             package='usb_cam',
    #             namespace=camera_name,
    #             name=camera_name,
    #             executable='usb_cam_node_exe',
    #             parameters=[{
    #             'framerate': 30.0,
    #             'image_width': 640,
    #             'image_height': 360,
    #             'pixel_format': 'yuyv',
    #             'camera_frame_id': 'usb_cam',
    #             'io_method': 'userptr',
    #             #  'exposure_auto': False, 
    #             #  'brightness': 128,
    #             # 'contrast': 32,
    #             # 'saturation': 64,
    #             # 'sharpness': 3, 
    #             'white_balance_temperature_auto':False,
    #             # 'focus_auto':False,
    #             #'autofocus': False,
    #             #'focus': 4,
    #             # 'autoexposure': True
    #         }],
    #             output='screen',
    #         ),
    #     )

    # realsense_ros_launch_includes_group_action = GroupAction(
    # #    condition=IfCondition(LaunchConfiguration('use_cameras')),
    # #    actions=rs_actions,
    #  )
    
    slate_base_node = Node(
        package='interbotix_slate_driver',
        executable='slate_base_node',
        name='slate_base',
        output='screen',
        namespace='mobile_base',
        condition=IfCondition(LaunchConfiguration('use_base')),
    )
    








    joystick_teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='base_joystick_teleop',
        namespace='mobile_base',
        parameters=[
            ParameterFile(
                PathJoinSubstitution([
                    FindPackageShare('aloha'),
                    'config',
                    'base_joystick_teleop.yaml'
                ]),
                allow_substs=True,
            ),
        ],
        condition=AndCondition([
            IfCondition(LaunchConfiguration('use_base')),
            IfCondition(LaunchConfiguration('use_joystick_teleop')),
        ]),
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        namespace='mobile_base',
        parameters=[{
            'dev': '/dev/input/js0',
            'deadzone': 0.3,
            'autorepeat_rate': 20.0,
        }],
        condition=AndCondition([
            IfCondition(LaunchConfiguration('use_base')),
            IfCondition(LaunchConfiguration('use_joystick_teleop')),
        ]),
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d', LaunchConfiguration('aloha_rvizconfig')
        ],
        condition=IfCondition(LaunchConfiguration('use_aloha_rviz')),
    )

    loginfo_action = LogInfo(msg=[
        '\nBringing up ALOHA with the following launch configurations: ',
        '\n- launch_leaders: ', LaunchConfiguration('launch_leaders'),
        '\n- use_cameras: ', LaunchConfiguration('use_cameras'),
        '\n- is_mobile: ', LaunchConfiguration('is_mobile'),
        '\n- use_base: ', LaunchConfiguration('use_base'),
        '\n- use_joystick_teleop: ', LaunchConfiguration('use_joystick_teleop'),
    ])

    #Create the gravity compensation nodes
    leader_left_gravity_compensation_node = Node(
        package='interbotix_gravity_compensation',
        executable='interbotix_gravity_compensation',
        name='gravity_compensation',
        namespace=LaunchConfiguration('robot_name_leader_left'),
        output='screen',
        emulate_tty=True,
        parameters=[{'motor_specs': LaunchConfiguration('leader_motor_specs_left')}],
        condition=IfCondition(LaunchConfiguration('use_gravity_compensation')),
    )

    leader_right_gravity_compensation_node = Node(
        package='interbotix_gravity_compensation',
        executable='interbotix_gravity_compensation',
        name='gravity_compensation',
        namespace=LaunchConfiguration('robot_name_leader_right'),
        output='screen',
        emulate_tty=True,
        parameters=[{'motor_specs': LaunchConfiguration('leader_motor_specs_right')}],
        condition=IfCondition(LaunchConfiguration('use_gravity_compensation')),
    )
    print("launched")
    return [
        xsarm_control_leader_left_launch_include,
        xsarm_control_leader_right_launch_include,
        xsarm_control_follower_left_launch_include,
        xsarm_control_follower_right_launch_include,
        leader_left_transform_broadcaster_node,
        leader_right_transform_broadcaster_node,
        follower_left_transform_broadcaster_node,
        follower_right_transform_broadcaster_node,
        #realsense_ros_launch_includes_group_action,
        slate_base_node,
        joystick_teleop_node,
        joy_node,
        rviz2_node,
        loginfo_action,
        leader_left_gravity_compensation_node,
        leader_right_gravity_compensation_node,
    ]


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_model_leader',
            default_value='aloha_wx250s',
            description='model type of the leader arms.'
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_model_follower',
            default_value='aloha_vx300s',
            description='model type of the follower arms.'
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_leader_left',
            default_value='leader_left',
            description='name of the left leader arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_leader_right',
            default_value='leader_right',
            description='name of the right leader arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_follower_left',
            default_value='follower_left',
            description='name of the left follower arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_follower_right',
            default_value='follower_right',
            description='name of the right follower arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_modes_left',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'leader_modes_left.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the left leader arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_modes_right',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'leader_modes_right.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the right leader arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'follower_modes_left',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'follower_modes_left.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the left follower arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'follower_modes_right',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'follower_modes_right.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the right follower arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'launch_leaders',
            default_value='true',
            choices=('true', 'false'),
            description=(
                'if `true`, launches both the leader and follower arms; if `false, just the '
                'followers are launched'
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_cameras',
            default_value='true',
            choices=('true', 'false'),
            description='if `true`, launches the camera drivers.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_high_name',
            default_value='cam_high',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_low_name',
            default_value='cam_low',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_left_wrist_name',
            default_value='cam_left_wrist',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_right_wrist_name',
            default_value='cam_right_wrist',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'is_mobile',
            default_value=EnvironmentVariable(
                name='INTERBOTIX_ALOHA_IS_MOBILE',
                default_value='true',
            ),
            choices=('true', 'false'),
            description='',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_base',
            default_value=LaunchConfiguration('is_mobile'),
            choices=('true', 'false'),
            description='if `true`, launches the driver for the SLATE base',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_joystick_teleop',
            default_value=LaunchConfiguration('use_base'),
            choices=('true', 'false'),
            description='if `true`, launches a joystick teleop node for the base',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_aloha_rviz',
            default_value='false',
            choices=('true', 'false'),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'aloha_rvizconfig',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'rviz',
                'aloha.rviz',
            ]),
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_launch_arguments(
            robot_description_launch_config_name='robot_description_leader_left',
            robot_model_launch_config_name='robot_model_leader',
            robot_name_launch_config_name='robot_name_leader_left',
            base_link_frame='base_link',
            use_world_frame='false',
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_launch_arguments(
            robot_description_launch_config_name='robot_description_leader_right',
            robot_model_launch_config_name='robot_model_leader',
            robot_name_launch_config_name='robot_name_leader_right',
            base_link_frame='base_link',
            use_world_frame='false',
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_launch_arguments(
            robot_description_launch_config_name='robot_description_follower_left',
            robot_model_launch_config_name='robot_model_follower',
            robot_name_launch_config_name='robot_name_follower_left',
            base_link_frame='base_link',
            use_world_frame='false',
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_launch_arguments(
            robot_description_launch_config_name='robot_description_follower_right',
            robot_model_launch_config_name='robot_model_follower',
            robot_name_launch_config_name='robot_name_follower_right',
            base_link_frame='base_link',
            use_world_frame='false',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_gravity_compensation',
            default_value='true',
            choices=('true', 'false'),
            description='if `true`, launches the gravity compensation node',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_motor_specs_left',
            default_value=[
                PathJoinSubstitution([
                    FindPackageShare('aloha'),
                    'config',
                    'leader_motor_specs_left.yaml'])
            ],
            description="the file path to the 'motor specs' YAML file for the left leader arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_motor_specs_right',
            default_value=[
                PathJoinSubstitution([
                    FindPackageShare('aloha'),
                    'config',
                    'leader_motor_specs_right.yaml'])
            ],
            description="the file path to the 'motor specs' YAML file for the right leader arm.",
        )
    )
    
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])