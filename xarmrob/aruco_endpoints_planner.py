#!/usr/bin/env python3

# ROS node to command an Endpoint to a HiWonder xArm 1S 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2024-11-12

import rclpy
from rclpy.node import Node 
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup 
from rclpy.executors import MultiThreadedExecutor
import threading
import numpy as np
import traceback
import time
# from sensor_msgs.msg import JointState
# from xarmrob_interfaces.srv import ME439XArmInverseKinematics #, ME439XArmForwardKinematics
from xarmrob_interfaces.msg import ME439PointXYZ

import xarmrob.smooth_interpolation as smoo 

from example_interfaces.msg import String
from sensor_msgs.msg import JointState

## Define a temporary function using Python "lambda" functionality to print colored text
# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal/3332860#3332860
# search that page for "CircuitSacul" to find that answer
coloredtext = lambda r, g, b, text: f'\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m'



class ArucoEndpointsPlanner(Node): 
    def __init__(self): 
        super().__init__('aruco_endpoints_planner')


        self.xyz_goal = [0.165, 0.0, 0.155] # roughly upright neutral with wrist at 45 degrees. Formally: [0.1646718829870224, 0.0, 0.1546700894832611]
        self.old_xyz_goal = [0.05, 0.0, 0.155]
        self.xyz_traj = [self.old_xyz_goal]
        self.disp_traj = []
        self.gripper = 0
        self.idx = 0

        # =============================================================================
        #   # Publisher for the Endpoint goal. 
        # =============================================================================
        self.pub_endpoint_desired = self.create_publisher(ME439PointXYZ,'/endpoint_desired',1,callback_group=ReentrantCallbackGroup())
        # Create the message, with a nominal pose
        self.endpoint_desired_msg = ME439PointXYZ()
        self.endpoint_desired_msg.xyz = self.xyz_goal 

        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 
        self.endpoint_speed = self.declare_parameter('endpoint_speed',0.05).value  # nominal speed for continuous movement among points. 
        

        # =============================================================================
        #   # new parameters
        # =============================================================================
        self.sub_aruco = self.create_subscription(String, '/aruco', self.aruco_msg_processor, 1)
        
        # self.sub_joint_angles = self.create_subscription(JointState, '/joint_angles_desired', self.joint_states_callback, 1)
        self.pub_joint_angles = self.create_publisher(JointState, '/joint_angles_desired', 1, callback_group=ReentrantCallbackGroup())
        self.pub_object_pos = self.create_publisher(ME439PointXYZ,'/object_pos',1,callback_group=ReentrantCallbackGroup())

        self.y_rotation_sign = np.sign(self.declare_parameter('y_rotation_sign',1).value)
        
        # Vectors from each frame origin to the next frame origin, in the proximal
        # convert parameters to numpy arrays with shape (3,1) for easier matrix multiplication (column vectors)
        self.r_01 = np.column_stack(self.declare_parameter('frame_offset_01',[0., 0., 0.074]).value).transpose()
        self.r_12 = np.column_stack(self.declare_parameter('frame_offset_12',[0.010, 0., 0.]).value).transpose()
        self.r_23 = np.column_stack(self.declare_parameter('frame_offset_23',[0.101, 0., 0.]).value).transpose()
        self.r_34 = np.column_stack(self.declare_parameter('frame_offset_34',[0.0627, 0., 0.0758]).value).transpose()        
        self.r_45 = np.column_stack(self.declare_parameter('frame_offset_45',[0., 0., 0.]).value).transpose()
        self.r_56 = np.column_stack(self.declare_parameter('frame_offset_56',[0., 0., 0.]).value).transpose()
        self.r_6cam = np.column_stack(self.declare_parameter('camera_offset',[0.117, -0.015, 0.037]).value).transpose()

        # initialize parameter
        self.id_to_position = {23: [0.0, -0.2, 0.1]}
        self.camera_updating = True
        self.object_dict = {} # {id: [x_world, y_world, z_world]}
        self.midpoint = self.declare_parameter('midpoint', [0.15, 0.0, 0.15]).value  # midpoint for the robot to move to before sorting
        # print object_dict
        # initialize the first position
        self.initial_joint_angles = self.declare_parameter('initial_joint_angles', [0., -2.152, 1.5707, 0., 1.419, 0., 0.]).value
        self.joint_angles_desired_msg = JointState()
        
        self.timer = self.create_timer(3.0, self.switch_to_sorting)

        self.pub_debug_msg = self.create_publisher(String, '/debug', 1)
        self.debug_msg = String()

    def switch_to_sorting(self):
        """
        Switch to the sorting mode, where the robot will sort the object.
        """
        if self.object_dict == {}:
            self.get_logger().info(coloredtext(255, 0, 0, "No objects detected. Cannot switch to sorting mode."))
            # TODO: BUG: 30% not posing correctly
            self.publish_joint_angles(self.initial_joint_angles)
            return
        self.get_logger().info(f"Using object dictionary : {self.object_dict}")

        self.camera_updating = False
        self.get_logger().info(coloredtext(0, 255, 0, "Switching to grasp mode."))
        
        self.start_sorting()

        self.timer.cancel()  # Cancel the timer to stop sending endpoint commands

    def start_sorting(self):
        for id, start in self.object_dict.items():
            start[0] -= 0.05
            start[2] = 0.09
            midpoint = self.midpoint
            dest = self.id_to_position.get(id, None)  # Default position if ID not found
            if dest is None:
                self.get_logger().warning(f"No destination found for ID: {id}")
                continue

            self.debug_msg.data = f"Sorting object {id} at: {start} -> {dest}"
            self.pub_debug_msg.publish(self.debug_msg)

            # Move above the object, then to the object, and finally to its destination
            self.move_to(self.midpoint)
            self.move_to(start)
            self.move_to(self.midpoint)
            self.move_to(dest)
            self.move_to(self.midpoint)

        self.get_logger().info("Finished sorting all detected objects")

    def set_gripper(self, angle):
        pass

    def hold_gripper(self):
        pass

    def release_gripper(self):
        pass

    def move_to(self, new_xyz_goal):
        """Create and execute a trajectory to the specified endpoint."""
        self.create_trajectory_from_endpoint(new_xyz_goal)
        while self.idx < len(self.disp_traj):
            self.send_endpoint_desired()
            time.sleep(self.movement_time_ms / 1000)
        self.old_xyz_goal = list(new_xyz_goal)
        self.debug_msg.data = f"Moving to: {new_xyz_goal}"
        self.pub_debug_msg.publish(self.debug_msg)


    # Callback to publish the endpoint at the specified rate. 
    def send_endpoint_desired(self):
        # print(self.idx)
        if self.disp_traj == []:
            # self.get_logger().info("No trajectory to follow. Waiting for new input.")
            return
        if self.idx>=len(self.disp_traj):
            self.idx = len(self.disp_traj) - 1
        # self.get_logger().info(f"Sending endpoint desired: {self.disp_traj[self.idx]}")
        self.xyz_goal = self.disp_traj[self.idx]
        self.idx += 1
        self.endpoint_desired_msg.xyz = self.xyz_goal 
        self.pub_endpoint_desired.publish(self.endpoint_desired_msg)

    def create_trajectory_from_endpoint(self, new_xyz_goal):
        # Do linear or minimum jerk interpolation
        self.t,self.disp_traj = smoo.minimum_jerk_interpolation(np.array(self.old_xyz_goal), np.array(new_xyz_goal), self.endpoint_speed, self.command_frequency)
        
        # Reset counter and wait until the trajectory has been played
        self.idx = 0
        # while self.idx < len(self.disp_traj):
        #     time.sleep(0.1)

    def joint_states_callback(self, joint_states_msg: JointState):
        self.joint_angles = joint_states_msg.position  # Get the joint angles from the message

    def aruco_msg_decoder(self, aruco_msg: String):
        """
        Decode the String message from the ARUCO tracker.
        Returns a dictionary {id: {x_world:, y_world: float, z_world: float}}
        """
        decoded_data = {}
        parts = aruco_msg.data.split('ID:')
        # print("parts", parts)
        for part in parts:
            values = part.strip().split(',')
            if len(values) == 7:
                id_ = int(values[0])
                # id_ = 'data' # TODO: TEMP FIX 
            
                coords = {'x': float(values[3]), 'y': -1 * float(values[1]), 'z': -1 * float(values[2])}
                # coords = tuple(float(v) for v in values[1:4])
                # rot_vec = tuple(float(v) for v in values[4:7])
                # decoded_data[id_] = (coords, rot_vec)
                decoded_data[id_] = coords
            else:
                print(f"Skipping malformed part: {part}")
        # print(decoded_data)
        self.aruco_msg_decoded = decoded_data
        # print(f"Decoded ARUCO message: {decoded_data}")
        return decoded_data

    def transform_to_world_coordinates(self, aruco_msg, joint_angles: list):
        # print(f"Transforming ARUCO message: {aruco_msg}")
        # print(type(aruco_msg))
        aruco_msg_decoded = self.aruco_msg_decoder(aruco_msg)
        # Unpack joint angles.
        alpha0 = joint_angles[0]
        beta1 = joint_angles[1] * self.y_rotation_sign
        beta2 = joint_angles[2] * self.y_rotation_sign
        gamma3 = joint_angles[3]
        beta4 = joint_angles[4] * self.y_rotation_sign
        gamma5 = joint_angles[5]    
        
        # =============================================================================
        # # Transformation from frame 0 to 1 (+Z axis rotation)
        # =============================================================================
        # "Rotation matrix of frame 1 in frame 0's coordinates" (columns are unit vectors of Frame 1 in Frame 0 coordinates)
        R_01 = np.array([ [np.cos(alpha0), -np.sin(alpha0), 0.], 
                             [np.sin(alpha0), np.cos(alpha0), 0.],
                             [       0.,           0.,  1.] ])
        # "Homogeneous Transform of Frame 1 in Frame 0's Coordinates"
        T_01 = np.vstack( (np.column_stack( (R_01, self.r_01) ) , [0., 0., 0., 1.]) )
        
        # =============================================================================
        # # Transformation from frame 1 to 2 (+Y axis rotation)
        # =============================================================================
        R_12 = np.array([ [ np.cos(beta1), 0., np.sin(beta1)], 
                           [       0. ,     1.,        0.    ],
                           [-np.sin(beta1), 0., np.cos(beta1)] ])
        T_12 = np.vstack( (np.column_stack( (R_12, self.r_12) ) , [0., 0., 0., 1.]) )
            
        # =============================================================================
        # # Transformation from frame 2 to 3 (+Y axis rotation)
        # =============================================================================
        R_23 = np.array([ [ np.cos(beta2), 0., np.sin(beta2)], 
                           [       0. ,     1.,        0.    ],
                           [-np.sin(beta2), 0., np.cos(beta2)] ])
        T_23 = np.vstack( (np.column_stack( (R_23, self.r_23) ) , [0., 0., 0., 1.]) )
            
        # =============================================================================
        # # Transformation from frame 3 to 4 (+X axis rotation)
        # =============================================================================
        R_34 = np.array([ [ 1. ,        0.     ,        0.      ], 
                           [ 0. , np.cos(gamma3), -np.sin(gamma3)], 
                           [ 0. , np.sin(gamma3),  np.cos(gamma3)] ])
        T_34 = np.vstack( (np.column_stack( (R_34, self.r_34) ) , [0., 0., 0., 1.]) )
                
        # =============================================================================
        # # Transformation from frame 4 to 5 (+Y axis rotation)
        # =============================================================================
        R_45 = np.array([ [ np.cos(beta4), 0., np.sin(beta4)], 
                           [       0. ,     1.,        0.    ],
                           [-np.sin(beta4), 0., np.cos(beta4)] ])
        T_45 = np.vstack( (np.column_stack( (R_45, self.r_45) ) , [0., 0., 0., 1.]) )
                
        # =============================================================================
        # # Transformation from frame 5 to 6 (+X axis rotation)
        # =============================================================================
        R_56 = np.array([ [ 1. ,        0.     ,        0.      ], 
                           [ 0. , np.cos(gamma5), -np.sin(gamma5)], 
                           [ 0. , np.sin(gamma5),  np.cos(gamma5)] ])
        T_56 = np.vstack( (np.column_stack( (R_56, self.r_56) ) , [0., 0., 0., 1.]) )
        
        # return T_01, T_12, T_23, T_34, T_45, T_56         

        # Vector of Zero from the frame origin in question, augmented with a 1 so it can be used with the Homogeneous Transform        
        T_02 = T_01@T_12
        T_03 = T_02@T_23
        T_04 = T_03@T_34
        T_05 = T_04@T_45
        T_06 = T_05@T_56


        for id_ in aruco_msg_decoded:
            print(f"Processing ARUCO ID: {id_}")
            aruco_pos = np.array((aruco_msg_decoded[id_]['x'], aruco_msg_decoded[id_]['y'], aruco_msg_decoded[id_]['z'])).reshape((3,1))  # Convert to a column vector
            # print(aruco_pos)
            r_object = aruco_pos + self.r_6cam  # Add the camera offset to the ARUCO position
            # print(r_object)        
            pos_endpoint = (T_06 @ np.vstack((r_object, 1)))[0:3,0]
            self.object_dict[id_] = pos_endpoint.tolist()

        
        # zero_stacked = np.array([0.06, -0.015, 0.037, 1]).reshape(4, 1)
        # p0 = (T_06 @ zero_stacked)[0:3,0]
        # return r_object.flatten().tolist()  # Convert to a list for easier handling in ROS messages
        # return p0

    def publish_joint_angles(self, joint_angles):
        self.joint_angles_desired_msg.position = np.array(joint_angles)
        self.joint_angles_desired_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_joint_angles.publish(self.joint_angles_desired_msg)
        

    def aruco_msg_processor(self, aruco_msg):
        """
        TODO:
        Take the string message, convert it, and then translate it to   the 
        endpoint in world coordinates from the camera's coordinate.
        add the aruco_msg to the offset position of the camera, and then apply the trasition matrixs to get the position of the camera
        """
        # if not self.camera_updating:
        #     return
        
        joint_angles = self.initial_joint_angles  # Get the joint angles from the class attribute
        self.transform_to_world_coordinates(aruco_msg, joint_angles)

    

def main(args=None):
    try: 
        # rclpy.init(args=args)
        # endpoint_keyboard_instance = EndpointKeyboard()  
        # thrd = threading.Thread(target=endpoint_keyboard_instance.endpoint_requests())
        # thrd.start()
        # # No need to "rclpy.spin(endpoint_keyboard_instance)" here because there's a while() loop blocking and keeping it alive. 
        # executor=MultiThreadedExecutor()
        # rclpy.spin(endpoint_keyboard_instance,executor=executor)
        # # rclpy.spin(endpoint_keyboard_instance)

        rclpy.init(args=args)
        aruco_endpoints_planner_instance = ArucoEndpointsPlanner()
        # executor = MultiThreadedExecutor()
        rclpy.spin(aruco_endpoints_planner_instance)
        
    except: 
        traceback.print_exc(limit=1)
        


if __name__ == '__main__':
    main()


