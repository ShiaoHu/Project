import numpy as np
from omni.isaac.cortex.df import DfNetwork, DfState, DfStateMachineDecider, DfStateSequence
from omni.isaac.cortex.dfb import DfBasicContext, DfGoTarget
from omni.isaac.cortex.motion_commander import PosePq
from omni.isaac.cortex.motion_commander import MotionCommand, ApproachParams, PosePq
from omni.isaac.core.utils.math import normalized
import sys
# Note that this is not the system level rospy, but one compiled for omniverse
import numpy as np
import rospy
#from cortex_control.msg import ButtonEvent
from std_msgs.msg import Empty, Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from robotiq_85_msgs.msg import GripperCmd 
from typing import Tuple
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
sys.path.append("/home/csi/ws_moveit/devel/lib/python3/dist-packages")
from yolo_color_detection.msg import ObjectInfo


import time

#import omni.isaac.cortex_sync.ros_tf_util as ros_tf_util
import omni.isaac.cortex.math_util as math_util






class ReachState(DfState):

    def __init__(self):
        self.target_p = np.array([0.6, 0.3, 0.6])
        self.target_o = np.array([0.0, -0.7071, 0.7071, 0.0])

        self.target_pose = PosePq(self.target_p, self.target_o)

        
        self.width = 0
        self.width_last = 0
        self.width_counter = 0

        self.ros_sub_1 = rospy.Subscriber("/phantom_right/phantom/pose", PoseStamped, self.Pose_callback, queue_size=100)
        self.ros_sub_2 = rospy.Subscriber("/phantom_right/phantom/button_grey", Int32, self.greyButt_callback, queue_size=1)
        self.ros_sub_3 = rospy.Subscriber("/phantom_right/phantom/button_white", Int32, self.whiteButt_callback, queue_size=1)
        self.ros_sub_4 = rospy.Subscriber("/gripper/joint_states", JointState, self.gripper_state_callback, queue_size=1)
        self.gripper_pub = rospy.Publisher("/gripper/cmd", GripperCmd, queue_size=10,latch=True)
        self.object_pose = None
        self.object_color = ""
        self.object_label = ""
        self.ros_sub_objinfo = rospy.Subscriber("/object_in_base", ObjectInfo, self.object_info_callback, queue_size=10)
        print("initialized")
        
        self.grey_stat = 0
        self.white_stat = 0
        self.end_effector_pose = 0
        self.target = 0
        #self.manual_control_enabled = True
        self.state = 0
        #self.button_state = 0
        

        self.first_selected = None
        self.P_point = np.array([0.136, -0.2, 0.05], dtype=np.float32)  # New target points for placing different objects
        self.A_point = np.array([0.336, -0.2, 0.05], dtype=np.float32)
        
        #change
        self.selected_object = None  # The first manually captured target type (Shape + Colour)
        self.detected_objects = []       # Automated grabbing of queues containing similar target locations
        
    
    def enter(self):
          
        self.context.robot.gripper.move_to(width = self.width*2, speed = 2)
        self.end_effector_pose = self.context.robot.arm.get_fk_p()


    def posestamp_to_T(self, pose_stamp: PoseStamped):
        p_msg = pose_stamp.pose.position
        p = np.array([p_msg.x, p_msg.y, p_msg.z])
        q_msg = pose_stamp.pose.orientation
        q = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z])
        T = math_util.pq2T(p, q)
        return p, q, T

    
    def Pose_callback(self, Pose_stat):

        if self.state == 0:
            position, quaternion, T = self.posestamp_to_T(Pose_stat)
            tran = np.array([0, 0, 0])

            rotation_matrix_z = np.array([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]])

            T_rot_z = math_util.pack_Rp(rotation_matrix_z, tran)
            T_final = np.dot(T_rot_z, T)

            p, q = math_util.T2pq(T_final)

            self.target_p[0] = p[0] * 1.0 + 0.45
            self.target_p[1] = p[1] * 1.3
            self.target_p[2] = p[2] * 1.5 + 0.2

            self.target_o = np.array([-0, 0.747, -0.747, 0])
            self.target_pose = PosePq(self.target_p, self.target_o)
            self.state = 0
        

        if self.white_stat:
            self.white_stat = 0
            for obj in self.detected_objects:
                if self._is_near(obj["position"], self.target_p):
                    self.selected_object = {"color": obj["color"], "label": obj["label"]}
                    print(f"[INFO] Recorded objects：color={obj['color']}，label={obj['label']}")
                    break
        
        if self.grey_stat:
            self.grey_stat = 0
            self.target = self.context.robot.arm.get_fk_p()
            self.state = 1

    def object_info_callback(self, msg):
        p = msg.position
        self.object_pose = [p.x, p.y, p.z]
        self.object_color = msg.color.lower()
        self.object_label = msg.label.lower()
        self._record_object()

    def _record_object(self):
        if self.object_pose is not None and self.object_color and self.object_label:
            self.detected_objects.append({
            "position": np.array(self.object_pose),  #change over to numpy array
            "color": self.object_color,
            "label": self.object_label
        })
        if len(self.detected_objects) > 50:
            self.detected_objects = self.detected_objects[-50:]

    def greyButt_callback(self, msg):
        if msg.data == 1 and self.grey_last == 0:
            self.grey_stat = 1
            print("[DEBUG] Grey key edge trigger successful")
        else:
            self.grey_stat = 0
        self.grey_last = msg.data

    def whiteButt_callback(self, msg):
        self.white_stat = msg.data

    def gripper_state_callback(self,msg):
        self.width = msg.position[0]
        self.width_counter = self.width_counter + 1
        if self.width_counter >= 15:
            self.width_last = self.width
            self.width_counter = 0

    def _is_near(self, pos1, pos2, threshold=0.05):
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) < threshold

    def _is_similar(self, obj1, obj2):
        try:
            result = obj1.get("color") == obj2.get("color") and obj1.get("label") == obj2.get("label")
            print(f"[DEBUG] comparisons: {obj1} vs {obj2} → {result}")
            return result
        except Exception as e:
            print(f"[WARN] _is_similar relatively unsuccessful: {e}, obj1={obj1}, obj2={obj2}")
            return False


    def step(self):
        self.end_effector_pose = self.context.robot.arm.get_fk_p()
        if self.state == 0:
            self.context.robot.arm.send_end_effector(target_pose=self.target_pose)
            if np.linalg.norm(self.target_p - self.end_effector_pose) < 1000:
                return None
            return self
        
        #move to object up
        if self.state == 2:
            current_pose = self.context.robot.arm.get_fk_p()
            self.D = np.linalg.norm(current_pose - self.up_point)
            if self.D > 0.06:
                print("[DEBUG] Current arm position1：", self.context.robot.arm.get_fk_p())
                print("[DEBUG] target grabbing location1：", self.up_point)
                print(self.D)
                self.target_pose = PosePq(self.up_point,self.target_o)
                self.context.robot.arm.send_end_effector(target_pose=self.target_pose)
            else:
                print("to4")
                rospy.sleep(5.0)
                self.state = 3
                return self
        
        #move to object
        if self.state == 3:
            current_pose = self.context.robot.arm.get_fk_p()
            y = abs(np.linalg.norm(current_pose[1] - self.object[1]))
            z = abs(np.linalg.norm(current_pose[2] - self.object[2]))
            print(z)
            self.DP = np.linalg.norm(current_pose - self.object)
            if self.DP > 0.06 or z > 0.05 or y > 0.03:
                print("[DEBUG] Current arm position 2：", self.context.robot.arm.get_fk_p())
                print("[DEBUG] Target grabbing Position 2：", self.object)
                print(self.DP)
                self.target_pose = PosePq(self.object,self.target_o)
                self.context.robot.arm.send_end_effector(target_pose=self.target_pose)
            else:
                #pick
                rospy.sleep(2.0)
                msg = GripperCmd()
                msg.position = 0
                msg.speed = 0.05
                msg.force = 100
                self.gripper_pub.publish(msg)
                rospy.sleep(3.0)
                print("to4")
                self.state = 4
                return self
            return self
        
        #move to object up again
        if self.state == 4:
            current_pose = self.context.robot.arm.get_fk_p()
            self.D = np.linalg.norm(current_pose - self.up_point)
            if self.D > 0.06:
                print("[DEBUG] Current arm position 3：", self.context.robot.arm.get_fk_p())
                print("[DEBUG] Target grabbing Position 3：", self.up_point)
                print(self.D)
                self.target_pose = PosePq(self.up_point,self.target_o)
                self.context.robot.arm.send_end_effector(target_pose=self.target_pose)
            else:
                print("to5")
                rospy.sleep(2.0)
                self.state = 5
                return self

        #move to target
        if self.state == 5:
            current_pose = self.context.robot.arm.get_fk_p()
            self.DA = np.linalg.norm(current_pose - self.target)
            self.target_pose = PosePq(self.target, self.target_o)
            if self.DA > 0.05:
                self.context.robot.arm.send_end_effector(target_pose=self.target_pose)
                print("[DEBUG] Current arm position 4：", self.context.robot.arm.get_fk_p())
                print("[DEBUG] Target put location：", self.target)
                print(self.DA)
                return self
            else:
                #drop
                rospy.sleep(2.0)
                msg = GripperCmd()
                msg.position = 0.085
                msg.speed = 0.05
                msg.force = 100
                self.gripper_pub.publish(msg)
                rospy.sleep(3.0)
                self.state = 1
                return self

        if self.state == 1:
            if self.selected_object is None:
                print("[WARN] Please use the white keys to record an object first")
                msg = GripperCmd()
                msg.position = 0.085
                msg.speed = 0.05
                msg.force = 100
                self.gripper_pub.publish(msg)

                self.state = 0
            else:
                nearest_obj = None
                min_dist = float("inf")
                for obj in self.detected_objects:
                    if self._is_similar(obj, self.selected_object):
                        dist = np.linalg.norm(self.context.robot.arm.get_fk_p() - obj["position"])
                        if dist < min_dist:
                            min_dist = dist
                            nearest_obj = obj

                if nearest_obj:
                    print(f"[INFO] Find the nearest matching object：color={nearest_obj['color']}，shape={nearest_obj['label']} - position={nearest_obj['position']}")
                    try:
                        self.position_np = np.array(nearest_obj["position"], dtype=np.float32).flatten()
                        self.up_point = np.copy(self.position_np)
                        self.up_point[2] += 0.15  # Directly above the object 10cm
                        self.object = np.copy(self.position_np)
                        self.object[1] += 0.05
                        self.A_point = np.array([0.336, -0.2, 0.05], dtype=np.float32)
                        current_pose = self.context.robot.arm.get_fk_p()

                        rospy.sleep(5.0)
                        self.state = 2
                        print("to2") 

                    except Exception as e:
                        print(f"[ERROR] The capture process has gone wrong.: {e}")


def make_decider_network(robot):
    reachstate = ReachState()
    root = DfStateMachineDecider(DfStateSequence([reachstate], loop=True))
    return DfNetwork(root, context=DfBasicContext(robot))
