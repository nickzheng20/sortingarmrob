from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Find the URDF file for the robot and get a robot description from it. 
    urdf_file_name = 'robot-xarm.urdf'
    urdf = os.path.join(
        get_package_share_directory('xarmrob'),
        'urdf',
        urdf_file_name
        )
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()
        
    # Find the YAML file of parameters
    params_file_name = 'robot_xarm_info.yaml'
    params_file = os.path.join(
        get_package_share_directory('xarmrob'),
        'config',
        params_file_name
        )
    
    # Create the Launch Description
    launch_descr = LaunchDescription([
        Node(
            package='xarmrob',
            executable='command_xarm',
            name='command_xarm',
            parameters=[params_file]
        ),
        Node(
            package='xarmrob',
            executable='xarm_kinematics',
            name='xarm_kinematics',
            parameters=[params_file]
        ),
        Node(
            package='xarmrob',
            executable='aruco_endpoints_planner',
            name='aruco_endpoints_planner',
            output='screen',
            parameters=[params_file]
        ),
        Node(
            package='xarmrob',
            executable='aruco',
            name='aruco',
        )
    ])
    
    return launch_descr
    

