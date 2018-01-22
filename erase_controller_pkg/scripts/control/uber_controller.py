#!/usr/bin/env python
#Author: Ariel Anders
import numpy as np
import rospy
import roslib; 
roslib.load_manifest("gdm_arm_controller")
from pr2_controllers_msgs.msg import \
        JointTrajectoryAction, JointTrajectoryGoal, JointTrajectoryActionGoal,\
        SingleJointPositionAction, SingleJointPositionGoal, SingleJointPositionActionGoal,\
        PointHeadAction, PointHeadGoal, PointHeadActionGoal,\
        Pr2GripperCommandAction, Pr2GripperCommandGoal, Pr2GripperCommandActionGoal

from pr2_gripper_sensor_msgs.msg import *

from pr2_mechanism_msgs.srv import SwitchController, ListControllers
from trajectory_msgs.msg import JointTrajectoryPoint
from actionlib import SimpleActionClient as SAC
from gdm_arm_controller.msg  import gdm_msg_values as status
from gdm_arm_controller.srv  import *
#from gdm_arm_controller.msg import gdm_command, gdm_trajectory_point, gdm_state, GDMAction, GDMFeedback, GDMGoal, GDMResult, gdm_msg_values
from geometry_msgs.msg import Twist, Pose,Point,Quaternion, PoseStamped, WrenchStamped, Vector3Stamped
from sensor_msgs.msg import JointState
import tf
from convert_functions import *
from actionlib_msgs.msg import GoalStatus as gs
from jtv_controller.msg import JTVCommand, JTVCartesianControllerState

" This allows you switch between different arm controllers"
class Pr2ControllerManager:

    def __init__(self):
        self.switch_controller_serv_name = 'pr2_controller_manager/switch_controller'
        list_controllers_serv_name = 'pr2_controller_manager/list_controllers'
        wait_for_service(self.switch_controller_serv_name)
        wait_for_service(list_controllers_serv_name)
        
        self.switch_controller_service = \
            rospy.ServiceProxy(self.switch_controller_serv_name, SwitchController)
        self.list_controllers_service = \
            rospy.ServiceProxy(list_controllers_serv_name, ListControllers)
        
        self.controllers = {
                'joint': "_arm_controller",
                'cartesian': "_cart"
                }


    def get_controllers_status(self, whicharm):
        resp = self.list_controllers_service()
        controllers = dict(zip(resp.controllers, resp.state))
        joint_name = whicharm + self.controllers["joint"]
        cart_name = whicharm + self.controllers["cartesian"]

        joint_loaded =  joint_name in controllers
        cart_loaded = cart_name in controllers
        if not cart_loaded:
            rospy.loginfo("Cartesian controller not loaded! error")
        joint_running = controllers[joint_name] =="running" \
            if joint_loaded else False
        cart_running = controllers[cart_name] =="running" \
            if cart_loaded else False
        rospy.loginfo("joint controller running: %s, cart controller running: %s "\
            %(joint_running, cart_running))
        
        return joint_running, cart_running



    def stop_controller(self, whicharm):

        ctrls  = [whicharm + self.controllers[ctrl] for ctrl in ['cartesian', 'joint'] ]
        wait_for_service(self.switch_controller_serv_name)
        self.switch_controller_service([], ctrls,2)
        """

        stop_name = whicharm + self.controllers[stop_ctrl]


        # Get current controllers status
        resp = self.list_controllers_service()
        controllers_status = dict(zip(resp.controllers, resp.state))

        # are controllers loaded?
        stop_loaded =  stop_name in controllers_status
        start_loaded =  start_name in controllers_status

        # are controllers running?
        stop_running = controllers_status[stop_name]=="running"\
                if stop_loaded else False  
        start_running = controllers_status[start_name]=="running"\
                if start_loaded else False  
        
        rospy.loginfo(("starting %s, stoppings %s for arm %s. Current status "\
                "of start, stop: %s, %s " % (start_ctrl, stop_ctrl, whicharm,\
                start_running, stop_running)))
        
        if start_running and not stop_running:
            rospy.loginfo("correct controllers laready running")
            pass
            # don't have to switch anything.  correct config
        else:
            wait_for_service(self.switch_controller_serv_name)
            rospy.sleep(.1)
            # turn start controller on and stop controller off
            if start_running and stop_running:
                self.switch_controller_service([], [stop_name],2)
            elif not start_running and not stop_running:
                self.switch_controller_service([start_name], [],2)
            elif not start_running and stop_running:
                self.switch_controller_service([start_name], [stop_name],2)

        """


    def start_controller(self, whicharm, start_ctrl):

        if not start_ctrl in ["cartesian", "joint"]:
                rospy.loginfo("start controller is cartesian or joint")
        stop_ctrl = "cartesian" if start_ctrl == "joint" else "joint"
        start_name = whicharm + self.controllers[start_ctrl]
        stop_name = whicharm + self.controllers[stop_ctrl]


        # Get current controllers status
        resp = self.list_controllers_service()
        controllers_status = dict(zip(resp.controllers, resp.state))

        # are controllers loaded?
        stop_loaded =  stop_name in controllers_status
        start_loaded =  start_name in controllers_status

        # are controllers running?
        stop_running = controllers_status[stop_name]=="running"\
                if stop_loaded else False  
        start_running = controllers_status[start_name]=="running"\
                if start_loaded else False  
        
        rospy.loginfo(("starting %s, stoppings %s for arm %s. Current status "\
                "of start, stop: %s, %s " % (start_ctrl, stop_ctrl, whicharm,\
                start_running, stop_running)))
        
        if start_running and not stop_running:
            rospy.loginfo("correct controllers laready running")
            pass
            # don't have to switch anything.  correct config
        else:
            wait_for_service(self.switch_controller_serv_name)
            rospy.sleep(.1)
            # turn start controller on and stop controller off
            if start_running and stop_running:
                self.switch_controller_service([], [stop_name],2)
            elif not start_running and not stop_running:
                self.switch_controller_service([start_name], [],2)
            elif not start_running and stop_running:
                self.switch_controller_service([start_name], [stop_name],2)

def wait_for_service(name, timeout=5.0):
    return
    while not rospy.is_shutdown():  
        rospy.loginfo("controller manager: waiting for %s to be there"%name)
        try:
            rospy.wait_for_service(name, timeout)
        except rospy.ROSException:
            continue
        break
    rospy.loginfo("controller manager: %s found"%name)  


class UberController:
    simple_clients = {
    'torso':(
        'torso_controller/position_joint_action',  
        SingleJointPositionAction),
    'head':(
        'head_traj_controller/point_head_action',
        PointHeadAction),
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
        PR2GripperEventDetectorAction),
    }
    


    #XXX do subscriber callback nicely

    """
    ===============================================================
                    Initializing all controllers    
    ===============================================================
    """ 


    def __init__(self):
        self.clients = {}
        self.joint_positions = {}
        self.tf_listener = tf.TransformListener()
        self.ctrl_mgr = Pr2ControllerManager()

        rospy.loginfo("starting simple action clients")
        for item in self.simple_clients:
            self.clients[item]= SAC(*self.simple_clients[item]) 
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
        self.cart_states = {}
        self.wrench_fk = {}
        self.accel = {}
        self.ft_sensor = None
        self.joint_sub = rospy.Subscriber("joint_states", JointState, self.jointCB)
        
        self.wrench_fk_sub ={
                'l': rospy.Subscriber("l_wrench_fk/tip_wrench_root_frame", WrenchStamped, self.l_wrench_fkCB),
                'r': rospy.Subscriber("r_wrench_fk/tip_wrench_root_frame", WrenchStamped, self.r_wrench_fkCB)
                }
        
        self.accel_sub ={
                'l': rospy.Subscriber("/l_accel_pub/accel_pub_base_frame", WrenchStamped, self.l_accelCB),
                'r': rospy.Subscriber("/r_accel_pub/accel_pub_base_frame", WrenchStamped, self.r_accelCB),
                }

        self.cart_pub = {
                'l' : rospy.Publisher("/l_cart/command_jtv", JTVCommand),
                'r' : rospy.Publisher("/r_cart/command_jtv", JTVCommand)
                }
        
        self.cart_sub = {
                'l' : rospy.Subscriber("/l_cart/state", JTVCartesianControllerState, self.l_cartCB),
                'r' : rospy.Subscriber("/r_cart/state", JTVCartesianControllerState, self.r_cartCB),
                }


        self.ft_sensor_sub = rospy.Subscriber("ft_pub", WrenchStamped, self.ft_sensorCB)
        self.gripper_events_sub= {}
        self.gripper_event = {}
        self.gripper_events_sub['l'] = rospy.Subscriber("/l_gripper_sensor_controller/event_detector_state", PR2GripperEventDetectorData, self.l_gripper_eventCB)
        self.gripper_events_sub['r'] = rospy.Subscriber("/r_gripper_sensor_controller/event_detector_state", PR2GripperEventDetectorData, self.r_gripper_eventCB)

        rospy.loginfo("ik service")
        self.call_ik = {}
        self.call_ik['l'] = rospy.ServiceProxy('l_get_ik', GetIK)
        self.call_ik['r'] = rospy.ServiceProxy('r_get_ik', GetIK)
        #rospy.wait_for_service("l_get_ik")
        #rospy.wait_for_service("r_get_ik")

        
        self.finished_registering()
        rospy.loginfo("done initializing Uber Controller!") 

    """ 
    ===============================================================
                    PR2 Controller Manager Services     
    ===============================================================
    """ 

    def start_joint(self, arm):
        self.ctrl_mgr.start_controller(arm, "joint")
        

    def start_cart(self, arm):
        self.ctrl_mgr.start_controller(arm, "cartesian")

    
    
    """ 
    ===============================================================
                    State subscriber callbacks    
    ===============================================================
    """ 

    def finished_registering(self):
        def not_ready():
            return (self.joint_positions.get('torso_lift_joint', None) == None)

        end_time = rospy.Time.now() + rospy.Duration (.5)
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
    
    def ft_sensorCB (self, data):
        self.ft_sensor = data
    
    def l_accelCB(self, data):
        self.accel['l'] = data
 
    def r_accelCB(self, data):
        self.accel['r'] = data

    def l_wrench_fkCB(self, data):
        self.wrench_fk['l'] = data
    
    def r_wrench_fkCB(self, data):
        self.wrench_fk['r'] = data

    def r_gripper_eventCB(self, data):
        self.gripper_event['r'] = data

    def l_gripper_eventCB(self, data):
        self.gripper_event['l'] = data

    def l_cartCB(self, data):
        self.cart_states['l'] = data
   
    def r_cartCB(self, data):
        self.cart_states['r'] = data

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
        msg = self.gripper_event[arm]
        event = msg.trigger_conditions_met or msg.acceleration_event
        return event

    def get_accel(self, arm):
        return self.accel[arm]

    def get_wrench_fk(self, arm):
        return self.cart_states[arm].eff
        #return self.wrench_fk[arm]

    def get_ft_sensor(self):
        return self.ft_sensor

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

        self.clients[client].send_goal(goal)
        rospy.sleep(.25)
        rospy.loginfo("command sent to %s client" % client)
        status = 0

        if blocking: #XXX why isn't this perfect?
            end_time = rospy.Time.now() + rospy.Duration(timeout+ .25)
            while (
                    (not rospy.is_shutdown()) and\
                    (rospy.Time.now() < end_time) and\
                    (status < gs.SUCCEEDED)):
                status =  self.clients[client].action_client.last_status_msg.status_list[-1].status
                self.rate.sleep()
            if status >gs.SUCCEEDED:
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
    
    def command_event_detector(self, arm, trigger, magnitude, blocking, timeout):
        goal = PR2GripperEventDetectorGoal()
        goal.command.trigger_conditions =  trigger
        goal.command.acceleration_trigger_magnitude=magnitude
        client = "%s_gripper_event"% arm
        return self.send_command(client, goal, blocking, timeout)

    def command_head(self, (x,y,z), frame, blocking, timeout): #pos is a tuple (x,y,z)
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(timeout)
        return self.send_command('head', goal, blocking, timeout)
    
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
        self.start_joint(arm)
        timeout=times[-1] + .5

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
        print joints
        print angles
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
    ===============================================================
           IK-Joint Trajectory Cartesian Control Commands     
    ===============================================================
    """ 
     # position is a list (x,y,z) (x,y,z,w) 
    def cmd_ik(self, arm, position, time, frame_id,blocking=True):
        #raw_input("move ik position?")
        # get ik
        for i in range(10):
            pose = list_to_pose_stamped(position, frame_id)
            result = self.call_ik[arm](pose)
            if result.succeeded.data:
                break
                pass
                #rospy.loginfo("IK call succeded: %s " % str(result) )
            else:
                rospy.loginfo("could not find IK solution")
                #return status.TIMEOUT_FAIL
       
        return self.command_joint_pose(arm, result.angles, time, blocking)

      # position is a list (x,y,z) (x,y,z,w) 
    def cmd_ik_interpolated(self, arm, (goal_pos,goal_rot), time, frame_id, blocking=True, use_cart=False, num_steps=30):
        angles = []
        times = []
        #self.freeze(arm)
        if use_cart: 
            self.start_cart(arm)
        init_pos, init_rot = self.return_cartesian_pose(arm, frame_id)
        
        for i in range(num_steps):
            fraction = (i+1.0)/float(num_steps)
            pose = interpolate_cartesian_step(
                            init_pos, init_rot,
                            goal_pos, goal_rot,
                            fraction)
            if use_cart: 
                self.cmd_cart(arm, pose, time, frame_id, blocking)
                rospy.sleep(float(time)/(num_steps))
            else:
                pose = list_to_pose_stamped(pose, frame_id)

                # if not using joint client start here
                result = self.call_ik[arm](pose)
                if result.succeeded.data:
                    pass
                    #rospy.loginfo("%d/%d IK call succeded: %s " % (i, num_steps, str(result) ))
                else:
                    rospy.loginfo("could not find IK solution.")
                    if i > 0: 
                        rospy.loginfo("completing partial execution")
                        continue
                    else:
                        return status.TIMEOUT_FAIL
                angles.append(result.angles)
                times.append(fraction*time)
        if not use_cart:
            rospy.loginfo("found interpolated cartesian ik trajectory")
            return self.command_joint_trajectory(arm, angles, times, blocking)
    
    """ 
    ===============================================================
                        Cartesian Control Commands     
    ===============================================================
    """ 
    def cmd_cart(self, arm, (goal_pos,goal_rot), time, frame_id, blocking=True):
        
        #self.start_cart(arm)
        cmd = JTVCommand()
        pose = PoseStamped()
        pose.pose.position = Point(*goal_pos)
        pose.pose.orientation = Quaternion(*goal_rot)
        pose.header.frame_id = frame_id
        pose.header.stamp = rospy.Time.now()
        
        cmd.x_desi = pose
        cmd.header = pose.header
        cmd.xd_desi.header = pose.header

        self.cart_pub[arm].publish(cmd)
        #raw_input("sent command!")
        #vel = self.cart_states[arm].xd
        
              
class Uber(UberController):
    timeout = 2

    def freeze(self, arm):
        self.start_joint(arm)
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
        self.command_head( (.1,0, 0), "base_link", blocking=blocking, timeout=self.timeout)

    def look_forward(self, blocking=True):
        self.command_head( (1,0, 1), "base_link", blocking=blocking, timeout=self.timeout)

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

    def move_arm_relative(self, arm, delta_pos, frame="base_link", time=2, blocking=True):
        (pos,rot) = self.return_cartesian_pose(arm, frame)
        new_pos = [x + y for (x,y) in zip(pos,delta_pos)]
        termination = status.TERMINATE_POSITION
        #return self.cmd_position(arm, (new_pos,rot), time, frame, termination, blocking)
        #return self.cmd_ik_interpolated(arm, (new_pos,rot), time, frame,  blocking)
        return self.cmd_ik(arm, (new_pos,rot), time, frame,  blocking)
   
    def guarded_position(self, arm, (desired_pos,rot), frame="base_link", time=5, use_z =False, use_cart=True):
        rate = rospy.Rate(100) # 10 hertz

        self.cmd_ik_interpolated(arm, (desired_pos, rot), time, frame, False, use_cart)
        if not use_cart:
            start_time = rospy.Time.now()
            end_time = start_time + rospy.Duration(time+.5)
            min_time = start_time + rospy.Duration(.1*time)
            i = 0

            while rospy.Time.now() < end_time and not rospy.is_shutdown():
                i += 1
                accel = self.get_wrench_fk(arm)
                x = accel.wrench.force.x
                y = accel.wrench.force.y
                z = accel.wrench.force.z
                
                if i % 5000 == 0:  print ("%.2f, %.2f, %.2f" %(x,y,z) )
                if rospy.Time.now() > min_time:
                    if abs(x) > 7:
                        rospy.loginfo("termination in x reached:  %s "% x)
                        break
                    elif abs(y) > 7:
                        rospy.loginfo("termination in y reached:  %s "% y)
                        break
                    elif use_z and abs(z) > 7:
                        rospy.loginfo("termination in z reached: %s " % z)
                        break
                    
           #XXX self.freeze(arm)


    def guarded_velocity(self, arm, velocity, frame="base_link", time=5, use_z =False):
        # execute command until time is reached
        #self.request_gripper_event(arm)
        rate = rospy.Rate(100) # 10 hertz
        start_pos, rot = self.return_cartesian_pose(arm, frame)
        desired_pos = [ start_pos[i] + velocity[i]*time for i in range(3) ]
        self.guarded_position(arm, (desired_pos,rot), frame, time, use_z, use_cart=True)

    def cmd_velocity(self, arm, velocity, frame="base_link", time=0.5):
        self.guarded_velocity(arm, velocity, frame, time, use_z=True)

    def move_arm_trajectory(self, arm, poses, frame):
        for (pose, t) in poses:
            self.cmd_ik_interpolated(arm, pose, t, frame, True)
    
    def push_down(self, arm):
        rospy.loginfo("PUSH DOWN")
        v = [0,0,-0.1]
        self.guarded_velocity('l', v, "base_link", time=.7)
    

if __name__=="__main__":

    def test_gripper():
        rospy.loginfo("testing open gripper commands")
        uc.close_gripper('l')
        uc.close_gripper('r')
        rospy.loginfo("grippers should be closed")
        uc.open_gripper('l')
        uc.open_gripper('r')
        rospy.loginfo("grippers should be open")
    
    def test_head():
        rospy.loginfo("testing head command")
        uc.look_down_center()
        raw_input("look up")
        uc.look_forward()
    
    def test_torso():
        rospy.loginfo("testing torso command")
        uc.lift_torso()
        raw_input("move torso down")
        uc.down_torso()

    def test_cart():
        rospy.loginfo("testing cartesian command")
        time = 5
        p = [0.5, .3, 0.4], [0,0,0,1]
        v = [1,0,0], [0,0,0]
        frame = "base_link"
        print uc.cmd_ik('l', p, time, frame)
        #print uc.cmd_ik_interpolated('l', p, time, frame)
        #print uc.cmd_position('l', p, time, frame)
        #raw_input("test velocity command")
        #print uc.cmd_velocity('l', v, time, frame)


    def test_joint():
        rospy.loginfo("testing joint control")
        uc.move_arm_to_side("l")
        uc.move_arm_to_side("r")

    def test_relative_cmds():
        rospy.loginfo("testing relative commands")
        uc.move_arm_relative('l', [0,0,-.1], time = 2)
        raw_input()
        uc.move_arm_relative('l', [0,0,.1], time= 2)
        raw_input()
        uc.move_arm_relative('l', [.1,0,0], time= 2 )
        raw_input()
        uc.move_arm_relative('l', [-.1,0,0], time=2 )

    def test_gripper_event(): 
        rospy.loginfo("requesting gripper event!-- right")
        print uc.wait_for_gripper_event('r')
        rospy.loginfo("requesting gripper event!-- left")
        uc.wait_for_gripper_event('l')

    def test_get_state():
        print "testing gathering state information"
        raw_input("get force torque sensor?")
        print uc.get_ft_sensor()

        raw_input("get wrench fk-- left")
        print uc.get_wrench_fk('l')

        raw_input("get wrench fk-- right")
        print uc.get_wrench_fk('r')

        raw_input("get joint angles-- left")
        print uc.get_joint_positions('l')

        raw_input("get joint angles-- right")
        print uc.get_joint_positions('r')
        uc.cmd_velocity('l', v)


        raw_input("get cartesian pose-- left")
        print uc.return_cartesian_pose('l', 'base_link')

        raw_input("get cartesian pose--right")
        print uc.return_cartesian_pose('r', 'base_link')

        
    def test_guarded_vel():
        delta = .05
        v = [0, 0,-delta]
        rospy.loginfo("testing guarded velocity +x")
        uc.guarded_velocity('l', v)
        rospy.loginfo("testing guarded velocity -x")
        delta = 0.025
        for i in range(10):
            v = [delta, 0,0]
            uc.cmd_velocity('l', v)
            v = [-delta, 0,0]
            uc.cmd_velocity('l', v)
            v = [0,delta,0]
            uc.cmd_velocity('l', v)
            v = [0,-delta,0]
            uc.cmd_velocity('l', v)


    rospy.init_node("ubertest")
    rospy.loginfo("how to use uber controller")
    uc = Uber()
    uc.command_torso(1, True,5)
    test_joint()
    test_head() 
    test_gripper()

    #test_guarded_vel()
    """
    test_head() 
    test_gripper_event()
    test_cart()
    test_torso()
    test_gripper()
    test_joint()
    test_relative_cmds()
    """
