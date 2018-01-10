#!/usr/bin/env python
#Author: Ariel Anders
import numpy as np
import rospy
import tf

from pr2_controllers_msgs.msg import \
        JointTrajectoryAction, JointTrajectoryGoal, JointTrajectoryActionGoal,\
        SingleJointPositionAction, SingleJointPositionGoal,\
        SingleJointPositionActionGoal,\
        PointHeadAction, PointHeadGoal, PointHeadActionGoal,\
        Pr2GripperCommandAction, Pr2GripperCommandGoal,\
        Pr2GripperCommandActionGoal
        

from pr2_gripper_sensor_msgs.msg import * #XXX fix

from pr2_mechanism_msgs.srv import SwitchController, ListControllers
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Twist
from actionlib import SimpleActionClient
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatus

class UberController:
    simple_clients = {
    'torso':(
        'torso_controller/position_joint_action',  
        SingleJointPositionAction),
    'head':('head_traj_controller/joint_trajectory_action',
	JointTrajectoryAction),
    'l_gripper': (
        'l_gripper_controller/gripper_action', 
        Pr2GripperCommandAction),
    'r_gripper': (
        'r_gripper_controller/gripper_action',
        Pr2GripperCommandAction),
    'r_joint': (
        'r_arm_controller/joint_trajectory_action',
        JointTrajectoryAction),
    'l_joint': (
        'l_arm_controller/joint_trajectory_action',
        JointTrajectoryAction ),
    'l_gripper_event':(
        'l_gripper_sensor_controller/event_detector',
        PR2GripperEventDetectorAction), 
    'r_gripper_event':(
        'r_gripper_sensor_controller/event_detector',
        PR2GripperEventDetectorAction)
    }

    #XXX do subscriber callback nicely

    """ 
    ============================================================================
                    Initializing all controllers    
    ============================================================================
    """ 


    def __init__(self):
        self.clients = {}
        self.joint_positions = {}
        self.tf_listener = tf.TransformListener()

        rospy.loginfo("starting simple action clients")
        for item in self.simple_clients:
            self.clients[item]= SimpleActionClient(*self.simple_clients[item]) 
            rospy.loginfo("%s client started" % item )
        
        for item in self.clients:
            res = self.clients[item].wait_for_server(rospy.Duration(.1))
            if res:
                rospy.loginfo("%s done initializing" % item )
            else:
                rospy.loginfo("Failed to start %s" % item )
        
            
        self.rate = rospy.Rate(10)
        rospy.loginfo("subscribing to state messages")
        
        self.joint_positons = {}
        self.joint_velocities = {}
        self.joint_sub = rospy.Subscriber(\
                "joint_states", JointState, self.jointCB)
        
        self.gripper_events_sub= {}
        self.gripper_event = {}
        self.gripper_events_sub['l'] = rospy.Subscriber(\
                "/l_gripper_sensor_controller/event_detector_state",\
                PR2GripperEventDetectorData, self.l_gripper_eventCB)
        self.gripper_events_sub['r'] = rospy.Subscriber(\
                "/r_gripper_sensor_controller/event_detector_state",\
                PR2GripperEventDetectorData, self.r_gripper_eventCB)
        
        self.finished_registering()
        rospy.loginfo("done initializing Uber Controller!") 

    """ 
    =============================================================== #XXX make these all nice :)
                    State subscriber callbacks    
    ===============================================================
    """ 

    def finished_registering(self):
        def not_ready():
            return (self.joint_positions.get('torso_lift_joint', None) == None)

        end_time = rospy.Time.now() + rospy.Duration (1)
        test = not_ready()
        while(not rospy.is_shutdown()and rospy.Time.now() <end_time and test):
            self.rate.sleep()
            test = not_ready()
        if not_ready():
            rospy.loginfo("Warning! did not complete subscribing") 
        else:
            rospy.loginfo("finished registering") 
    
    def jointCB(self,data):
        pos = dict(zip(data.name, data.position))
        vel = dict(zip(data.name, data.velocity))
        self.joint_positions = pos
        self.joint_velocities = vel 
    
    def r_gripper_eventCB(self, data):
        self.gripper_event['r'] = data

    def l_gripper_eventCB(self, data):
        self.gripper_event['l'] = data


    """ 
    ===============================================================
                   Get State information    
    ===============================================================
    """ 
 
    def get_torso_pose(self):
        return self.joint_positions['torso_lift_joint']

    def get_gripper_pose(self, arm):
        return self.joint_positions['%s_gripper_joint'%arm]

    def get_head_pose(self):
        head = (
                self.joint_positions['head_pan_joint'],
                self.joint_positions['head_tilt_joint'] )
        return head

    def get_gripper_event(self, arm):
        #This may not work until you subscribe to the gripper event 
        if arm in self.gripper_event:

            msg = self.gripper_event[arm]
            event = msg.trigger_conditions_met or msg.acceleration_event
            return event
        else:
            print "no gripper event found... did you launch gripper sensor action?"
            return None

    def get_accel(self, arm):
        return self.accel[arm]
        
    def get_joint_positions(self, arm):
        pos = [self.joint_positions[p] for p in self.get_joint_names(arm)]
        return pos
    
    def get_joint_velocities(self, arm):
        vel = [self.joint_velocities[p] for p in self.get_joint_names(arm)]
        return vel

    #return the current Cartesian pose of the gripper
    def return_cartesian_pose(self, arm, frame = 'base_link'):
        end_time = rospy.Time.now() + rospy.Duration(5)
        link = arm + "_gripper_tool_frame"
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            try:
                t = self.tf_listener.getLatestCommonTime(frame, link)
                (trans, rot) = self.tf_listener.lookupTransform(frame, link, t)
                return list(trans) , list(rot)
            except (tf.Exception, tf.ExtrapolationException):
                rospy.sleep(0.5)
                current_time = rospy.get_rostime()
                rospy.logerr(\
                "waiting for a tf transform between %s and %s"%\
                (frame, link))
        rospy.logerr("return_cartesian_pose waited 10 seconds tf\
                transform!  Returning None")
        return None, None

    """ 
    ===============================================================
                Send Commands for Action Clients                
    ===============================================================
    """ 
    def send_command(self, client, goal, blocking=False, timeout=None):
        if client == 'head':
            if blocking:
                self.clients[client].send_goal_and_wait(goal)
            else:
                self.clients[client].send_goal(goal)
        else:
            self.clients[client].send_goal(goal)
            rospy.sleep(.1)
            rospy.loginfo("command sent to %s client" % client)
            status = 0
            if blocking: #XXX why isn't this perfect?
                end_time = rospy.Time.now() + rospy.Duration(timeout+ .1)
                while (
                        (not rospy.is_shutdown()) and\
                        (rospy.Time.now() < end_time) and\
                        (status < GoalStatus.SUCCEEDED) and\
                        (type(self.clients[client].action_client.last_status_msg) != type(None))):
                    status = self.clients[client].action_client.last_status_msg.status_list[-1].status #XXX get to 80
                    self.rate.sleep()
                if status >= GoalStatus.SUCCEEDED:
                    rospy.loginfo("goal status achieved.  exiting")
                else:
                    rospy.loginfo("ending due to timeout")

            result = self.clients[client].get_result()
            return result
        return None

    def command_torso(self, pose, blocking, timeout):
        goal= SingleJointPositionGoal(
          position = pose, 
          min_duration=rospy.Duration(timeout),
          max_velocity=1.0)
        return self.send_command('torso', goal, blocking, timeout)
    
    def command_event_detector(self, arm, trigger, magnitude, blocking,timeout):
        goal = PR2GripperEventDetectorGoal()
        goal.command.trigger_conditions =  trigger
        goal.command.acceleration_trigger_magnitude=magnitude
        client = "%s_gripper_event"% arm
        return self.send_command(client, goal, blocking, timeout)

    def command_head(self, angles, time, blocking):
	    goal = JointTrajectoryGoal()
	    goal.trajectory.joint_names = ['head_pan_joint', 'head_tilt_joint']
	    point = JointTrajectoryPoint()
	    point.positions = angles
	    point.time_from_start = rospy.Duration(time)
	    goal.trajectory.points.append(point)
	    return self.send_command('head', goal, blocking, timeout=time)
    
    def command_gripper(self, arm, position, max_effort, blocking, timeout):
        goal = Pr2GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        client = "%s_gripper"% arm
        return self.send_command("%s_gripper"%arm, goal, blocking, timeout)
    
    """ 
    ===============================================================
                    Joint Control Commands 
    ===============================================================
    """ 
    # angles is a list of joint angles, times is a list of times from start
    def command_joint_trajectory(self, arm, angles, times, blocking):
        print angles
        timeout=times[-1] + 1.0

        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names =self.get_joint_names(arm)
        
        for (ang, t) in zip(angles, times):
            point = JointTrajectoryPoint()
            point.positions = ang
            point.time_from_start = rospy.Duration(t)
            goal.trajectory.points.append(point)
        goal.trajectory.header.stamp = rospy.Time.now()
        return self.send_command("%s_joint"%arm, goal, blocking, timeout)
    
    # for convience
    def command_joint_pose(self, arm, angles, time, blocking):
        #XXX test if at location
        joints = self.get_joint_positions(arm)
        raw = lambda x:( (x+np.pi) % 2*np.pi) - np.pi
        disp = sum( [abs( raw(x-y)) for (x,y) in zip (joints, angles) ])
        rospy.loginfo("displacement for joint pos is %s " % disp)
        if disp < 0.05:
            rospy.loginfo("already near position. not moving")
        else:
            return self.command_joint_trajectory(arm, [angles], [time],  blocking)

    def  get_joint_names (self, char):
        return [char+'_shoulder_pan_joint',
                char+'_shoulder_lift_joint',
                char+'_upper_arm_roll_joint',
                char+'_elbow_flex_joint',
                char+'_forearm_roll_joint',
                char+'_wrist_flex_joint',
                char+'_wrist_roll_joint' ]
     
""" 
============================================================================
            Uber builds on the UberController with set default values
            for easier use. 
============================================================================
""" 
             
class Uber(UberController):
    timeout = 2

    def freeze(self, arm):
        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names =self.get_joint_names(arm)
        goal.trajectory.header.stamp = rospy.Time.now()
        return self.send_command("%s_joint"%arm, goal, False)

    def set_timeout(self, time):
        self.timeout = time
    
    def wait_for_gripper_event(self,arm, timeout=None):
        if timeout == None: timeout = self.timeout
        #trigger = PR2GripperEventDetectorGoal().command.FINGER_SIDE_IMPACT_OR_ACC
        trigger = PR2GripperEventDetectorGoal().command.ACC
        #trigger = PR2GripperEventDetectorGoal().command.SLIP_AND_ACC
        #trigger = PR2GripperEventDetectorGoal().command.FINGER_SIDE_IMPACT_OR_SLIP_OR_ACC
        #trigger = PR2GripperEventDetectorGoal().command.SLIP
        magnitude = 4.0
        self.command_event_detector(arm, trigger, magnitude, True, timeout=timeout)
        return self.get_gripper_event(arm)

    def request_gripper_event(self, arm):
        trigger = PR2GripperEventDetectorGoal().command.ACC
        magnitude = 4.0
        self.command_event_detector(arm, trigger, magnitude, False, self.timeout)

    def open_gripper(self, arm, blocking=True):
        self.command_gripper(arm, 0.1, -1.0, blocking=blocking, timeout=self.timeout)
    
    def close_gripper(self,arm, blocking=True):
        self.command_gripper(arm, 0, 100, blocking=blocking, timeout=self.timeout)

    def look_down_center(self, blocking=True):
        self.command_head([0,np.pi/6.], 3, blocking=blocking)

    def look_forward(self, blocking=True):
        self.command_head([0,0], 3, blocking=blocking)

    def lift_torso(self, blocking=True):
        self.command_torso(0.2, blocking=blocking, timeout=self.timeout)
    
    def down_torso(self, blocking=True):
        self.command_torso(0.1, blocking=blocking, timeout=self.timeout)

    def move_arm_to_side(self, arm, blocking=True):
        l = [ 0.95, 0.0, np.pi/2., -np.pi/2., -np.pi*1.5, -np.pi/2.0, np.pi]
        r = [ -0.7, 0.0, -np.pi/2., -np.pi/2., np.pi*1.5, -np.pi/2.0, np.pi]
        if arm == "l":
            self.command_joint_pose('l', l, self.timeout, blocking=blocking)
        elif arm =="r":
            self.command_joint_pose('r', r, self.timeout, blocking=blocking)
