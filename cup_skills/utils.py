from __future__ import print_function
import math
import pdb
import os
import pickle
import platform
import pybullet as p
import numpy as np
import sys
import time
import cv2

from collections import defaultdict, deque, namedtuple
from cup_skills.local_setup import path
from itertools import product, combinations, count

from cup_skills.transformations import quaternion_from_matrix

directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(directory, '../motion'))
from motion_planners.rrt_connect import birrt, direct_path
#from ..motion.motion_planners.rrt_connect import birrt, direct_path

# from future_builtins import map, filter
# from builtins import input # TODO - use future
try:
   input = raw_input
except NameError:
   pass
user_input = input

INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI

#####################################

# Models

# Robots
DRAKE_IIWA_URDF = "models/drake/iiwa_description/urdf/iiwa14_polytope_collision.urdf"
KUKA_IIWA_URDF = "kuka_iiwa/model.urdf"
KUKA_IIWA_GRIPPER_SDF = "kuka_iiwa/kuka_with_gripper.sdf"
R2D2_URDF = "r2d2.urdf"
MINITAUR_URDF = "quadruped/minitaur.urdf"
HUMANOID_MJCF = "mjcf/humanoid.xml"
HUSKY_URDF = "husky/husky.urdf"

# Objects
KIVA_SHELF_SDF = "kiva_shelf/model.sdf"
SMALL_BLOCK_URDF = "models/drake/objects/block_for_pick_and_place.urdf"
BLOCK_URDF = "models/drake/objects/block_for_pick_and_place_mid_size.urdf"
SINK_URDF = 'models/sink.urdf'
STOVE_URDF = 'models/stove.urdf'

#####################################

# I/O

def is_darwin(): # TODO: change loading accordingly
    return platform.system() == 'Darwin' # platform.release()
    #return sys.platform == 'darwin'

def read(filename):
    with open(filename, 'r') as f:
        return f.read()

def write(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

def write_pickle(filename, data):  # NOTE - cannot pickle lambda or nested functions
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def safe_remove(p):
    if os.path.exists(p):
        os.remove(p)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

#####################################
#computer vision
#####################################
def threshold_img(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)
    #import ipdb; ipdb.set_trace()
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    gray =  cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    return gray


class Verbose(object):
    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.stdout = sys.stdout
            self.devnull = open(os.devnull, 'w')
            sys.stdout = self.devnull
        return self

    def __exit__(self, type, value, traceback):
        if not self.verbose:
            sys.stdout = self.stdout
            self.devnull.close()

# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/14797594#14797594
# https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262


class HideOutput(object):
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''
    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)

#####################################

# Savers

# TODO: move the saving to enter?

class Saver(object):
    def restore(self):
        raise NotImplementedError()
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        self.restore()

class ClientSaver(Saver):
    def __init__(self, new_client=None):
        self.client = CLIENT
        if new_client is not None:
            set_client(new_client)

    def restore(self):
        set_client(self.client)


class StateSaver(Saver):
    def __init__(self):
        self.state = save_state()

    def restore(self):
        restore_state(self.state)


#####################################

class PoseSaver(Saver):
    def __init__(self, body):
        self.body = body
        self.pose = get_pose(self.body)

    def restore(self):
        set_pose(self.body, self.pose)

class ConfSaver(Saver):
    def __init__(self, body): #, joints):
        self.body = body
        self.conf = get_configuration(body)

    def restore(self):
        set_configuration(self.body, self.conf)

#####################################

class BodySaver(Saver):
    def __init__(self, body): #, pose=None):
        #if pose is None:
        #    pose = get_pose(body)
        self.body = body
        self.pose_saver = PoseSaver(body)
        self.conf_saver = ConfSaver(body)

    def restore(self):
        self.pose_saver.restore()
        self.conf_saver.restore()

class WorldSaver(Saver):
    def __init__(self):
        self.body_savers = [BodySaver(body) for body in get_bodies()]

    def restore(self):
        for body_saver in self.body_savers:
            body_saver.restore()

#####################################

# Simulation

CLIENT = 0
# TODO: keep track of all the clients?

def get_client(client=None):
    if client is None:
        return CLIENT
    return client

def set_client(client):
    global CLIENT
    CLIENT = client

BODIES = defaultdict(dict)
# TODO: update delete as well

URDFInfo = namedtuple('URDFInfo', ['name', 'path'])

def load_pybullet(filename, fixed_base=False):
    body = p.loadURDF(filename, useFixedBase=fixed_base, physicsClientId=CLIENT)
    BODIES[CLIENT][body] = URDFInfo(None, filename)
    return body

URDF_FLAGS = [p.URDF_USE_INERTIA_FROM_FILE,
              p.URDF_USE_SELF_COLLISION,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS]

def load_model(rel_path, pose=None, fixed_base=True):
    # TODO: error with loadURDF when loading MESH visual and CYLINDER collision
    directory = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(directory, '..', rel_path)

    flags = 0 # by default, Bullet disables self-collision
    add_data_path()
    if abs_path.endswith('.urdf'):
        body = p.loadURDF(abs_path, useFixedBase=fixed_base, flags=flags, physicsClientId=CLIENT)
    elif abs_path.endswith('.sdf'):
        body = p.loadSDF(abs_path, physicsClientId=CLIENT)
    elif abs_path.endswith('.xml'):
        body = p.loadMJCF(abs_path, physicsClientId=CLIENT)
    elif abs_path.endswith('.bullet'):
        body = p.loadBullet(abs_path, physicsClientId=CLIENT)
    else:
        raise ValueError(abs_path)
    if pose is not None:
        set_pose(body, pose)
    BODIES[CLIENT][body] = URDFInfo(None, abs_path)
    return body

#####################################

def wait_for_duration(duration): #, dt=0):
    t0 = time.time()
    while (time.time() - t0) <= duration:
        disable_gravity()
    # TODO: wait until keypress

def simulate_for_duration(duration, dt=1/120.0):
    p.setTimeStep(dt)
    for i in range(int(duration/dt)):
        step_simulation()

def wait_for_interrupt(max_time=np.inf):
    """
    Hold Ctrl to move the camera as well as zoom
    """
    print('Press Ctrl-C to continue')
    try:
        wait_for_duration(max_time)
    except KeyboardInterrupt:
        pass
    finally:
        print()

# def wait_for_input(s=''):
#     print(s)
#     while True:
#         step_simulation()
#         line = sys.stdin.readline()
#         if line:
#             pass
#         #events = p.getKeyboardEvents() # TODO: only works when the viewer is in focus
#         #if events:
#         #    print(events)
#         # https://docs.python.org/2/library/select.html


def connect(use_gui=True, shadows=True):
    method = p.GUI if use_gui else p.DIRECT
    sim_id = p.connect(method)
    #sim_id = p.connect(p.GUI, options="--opengl2") if use_gui else p.connect(p.DIRECT)
    if use_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False, physicsClientId=sim_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, shadows, physicsClientId=sim_id)
    #visualizer_options = {
    #    p.COV_ENABLE_WIREFRAME: 1,
    #    p.COV_ENABLE_SHADOWS: 0,
    #    p.COV_ENABLE_RENDERING: 0,
    #    p.COV_ENABLE_TINY_RENDERER: 1,
    #    p.COV_ENABLE_RGB_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW: 0,
    #    p.COV_ENABLE_VR_RENDER_CONTROLLERS: 0,
    #    p.COV_ENABLE_VR_PICKING: 0,
    #    p.COV_ENABLE_VR_TELEPORTING: 0,
    #}
    #for pair in visualizer_options.items():
    #    p.configureDebugVisualizer(*pair)
    return sim_id

def disconnect():
    # TODO: change CLIENT?
    return p.disconnect(physicsClientId=CLIENT)

def is_connected():
    return p.getConnectionInfo(physicsClientId=CLIENT)['isConnected']

def get_connection(client=None):
    return p.getConnectionInfo(physicsClientId=get_client(client))['connectionMethod']

def has_gui(client=None):
    return get_connection(get_client(client)) == p.GUI

def get_data_path():
    import pybullet_data
    return pybullet_data.getDataPath()

def add_data_path():
    return p.setAdditionalSearchPath(get_data_path())

def enable_gravity(k=1):
    p.setGravity(0, 0, -9.8*k, physicsClientId=CLIENT)

def disable_gravity():
    p.setGravity(0, 0, 0, physicsClientId=CLIENT)

def step_simulation():
    p.stepSimulation(physicsClientId=CLIENT)

def enable_real_time():
    p.setRealTimeSimulation(1, physicsClientId=CLIENT)

def disable_real_time():
    p.setRealTimeSimulation(0, physicsClientId=CLIENT)

def update_state():
    # TODO: this doesn't seem to automatically update still
    disable_gravity()
    #step_simulation()
    #for body in get_bodies():
    #    for link in get_links(body):
    #        # if set to 1 (or True), the Cartesian world position/orientation
    #        # will be recomputed using forward kinematics.
    #        get_link_state(body, link)
    #for body in get_bodies():
    #    get_pose(body)
    #    for joint in get_joints(body):
    #        get_joint_position(body, joint)
    #p.getKeyboardEvents()
    #p.getMouseEvents()

def reset_simulation():
    p.resetSimulation(physicsClientId=CLIENT)

def get_camera():
    return p.getDebugVisualizerCamera(physicsClientId=CLIENT)

def set_camera(yaw, pitch, distance, target_position=np.zeros(3)):
    p.resetDebugVisualizerCamera(distance, yaw, pitch, target_position, physicsClientId=CLIENT)

def set_default_camera():
    set_camera(160, -35, 2.5, Point())

def save_state():
    return p.saveState(physicsClientId=CLIENT)

def restore_state(state_id):
    p.restoreState(stateId=state_id, clientServerId=CLIENT)

def save_bullet(filename):
    p.saveBullet(filename, physicsClientId=CLIENT)

def restore_bullet(filename):
    p.restoreState(fileName=filename, physicsClientId=CLIENT)

#####################################

# Geometry

def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return (point, quat_from_euler(euler))

def Pose2d(x=0., y=0., yaw=0.):
    return np.array([x, y, yaw])

def invert(pose):
    (point, quat) = pose
    return p.invertTransform(point, quat)

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose

def unit_from_theta(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)

def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat)

def unit_point():
    return (0., 0., 0.)

def unit_quat():
    return quat_from_euler([0, 0, 0]) # [X,Y,Z,W]

def unit_pose():
    return (unit_point(), unit_quat())

def get_length(vec):
    return np.linalg.norm(vec)

def angle_between(vec1, vec2):
    return np.math.acos(np.dot(vec1, vec2) / (get_length(vec1) *  get_length(vec2)))

def get_unit_vector(vec):
    norm = get_length(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm

def z_rotation(theta):
    return quat_from_euler([0, 0, theta])

def matrix_from_quat(quat):
    return p.getMatrixFromQuaternion(quat, physicsClientId=CLIENT)

def quat_from_matrix(mat):
    matrix = np.eye(4)
    matrix[:3,:3] = mat
    return quaternion_from_matrix(matrix)

def point_from_tform(tform):
    return np.array(tform)[:3,3]

def matrix_from_tform(tform):
    return np.array(tform)[:3,:3]

def point_from_pose(pose):
    return pose[0]

def quat_from_pose(pose):
    return pose[1]

def tform_from_pose(pose):
    (point, quat) = pose
    tform = np.eye(4)
    tform[:3,3] = point
    tform[:3,:3] = matrix_from_quat(quat)
    return tform

def pose_from_tform(tform):
    return point_from_tform(tform), quat_from_matrix(matrix_from_tform(tform))

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)

def base_values_from_pose(pose):
    (point, quat) = pose
    x, y, _ = point
    roll, pitch, yaw = euler_from_quat(quat)
    assert (abs(roll) < 1e-3) and (abs(pitch) < 1e-3)
    return (x, y, yaw)

def pose_from_base_values(base_values, default_pose):
    x, y, yaw = base_values
    _, _, z = default_pose[0]
    roll, pitch, _ = euler_from_quat(default_pose[1])
    return (x, y, z), quat_from_euler([roll, pitch, yaw])

#####################################

# Bodies

def get_bodies():
    return [p.getBodyUniqueId(i, physicsClientId=CLIENT)
            for i in range(p.getNumBodies(physicsClientId=CLIENT))]

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=CLIENT))

def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')

def get_body_name(body):
    return get_body_info(body).body_name.decode(encoding='UTF-8')

def get_name(body):
    name = get_body_name(body)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))

def has_body(name):
    try:
        body_from_name(name)
    except ValueError:
        return False
    return True

def body_from_name(name):
    for body in get_bodies():
        if get_body_name(body) == name:
            return body
    raise ValueError(name)

def remove_body(body):
    return p.removeBody(body, physicsClientId=CLIENT)

def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)
    #return np.concatenate([point, quat])

def get_point(body):
    return get_pose(body)[0]

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_base_values(body):
    return base_values_from_pose(get_pose(body))

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)

def set_point(body, point):
    set_pose(body, (point, get_quat(body)))

def set_quat(body, quat):
    set_pose(body, (get_point(body), quat))

def set_base_values(body, values):
    _, _, z = get_point(body)
    x, y, theta = values
    set_point(body, (x, y, z))
    set_quat(body, z_rotation(theta))

def is_rigid_body(body):
    for joint in get_joints(body):
        if is_movable(body, joint):
            return False
    return True

def is_fixed_base(body):
    return get_mass(body) == STATIC_MASS

def dump_body(body):
    print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
        body, get_body_name(body), is_rigid_body(body), is_fixed_base(body)))
    for joint in get_joints(body):
        print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Limits: {}'.format(
            joint, get_joint_name(body, joint), JOINT_TYPES[get_joint_type(body, joint)],
            is_circular(body, joint), get_joint_limits(body, joint)))
    print('Link id: {} | Name: {} | Mass: {}'.format(-1, get_base_name(body), get_mass(body)))
    for link in get_links(body):
        print('Link id: {} | Name: {} | Parent: {} | Mass: {}'.format(
            link, get_link_name(body, link), get_link_name(body, get_link_parent(body, link)),
            get_mass(body, link)))
        #print(get_joint_parent_frame(body, link))
        #print(map(get_data_geometry, get_visual_data(body, link)))
        #print(map(get_data_geometry, get_collision_data(body, link)))

def dump_world():
    for body in get_bodies():
        dump_body(body)
        print()

#####################################

# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute', # 0
    p.JOINT_PRISMATIC: 'prismatic', # 1
    p.JOINT_SPHERICAL: 'spherical', # 2
    p.JOINT_PLANAR: 'planar', # 3
    p.JOINT_FIXED: 'fixed', # 4
    p.JOINT_POINT2POINT: 'point2point', # 5
    p.JOINT_GEAR: 'gear', # 6
}

def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_joints(body):
    return list(range(get_num_joints(body)))

def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=CLIENT))

def get_joint_name(body, joint):
    return get_joint_info(body, joint).jointName.decode('UTF-8')

#def get_joint_names(body):
#    return [get_joint_name(body, joint) for joint in get_joints(body)]

def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)

def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True

def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)

JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                     'jointReactionForces', 'appliedJointMotorTorque'])

def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint, physicsClientId=CLIENT))

def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition

def get_joint_torque(body, joint):
    return get_joint_state(body, joint).appliedJointMotorTorque

def get_joint_positions(body, joints=None):
    return tuple(get_joint_position(body, joint) for joint in joints)

def set_joint_position(body, joint, value):
    try:
        p.resetJointState(body, joint, value, physicsClientId=CLIENT)
    except TypeError:
        if joint == 0:
            #I've learned not to question these things
            p.resetJointState(body, joint, value, physicsClientId=CLIENT)

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    #for joint, value in zip(joints, values):
    for i in range(len(joints)):
        joint = joints[i]
        value = values[i]
        set_joint_position(body, joint, value)

def get_configuration(body):
    return get_joint_positions(body, get_movable_joints(body))

def set_configuration(body, values):
    set_joint_positions(body, get_movable_joints(body), values)

def get_full_configuration(body):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body))

def get_joint_type(body, joint):
    return get_joint_info(body, joint).jointType

def is_movable(body, joint):
    return get_joint_type(body, joint) != p.JOINT_FIXED

def get_movable_joints(body): # 45 / 87 on pr2
    return [joint for joint in get_joints(body) if is_movable(body, joint)]

def joint_from_movable(body, index):
    return get_joints(body)[index]

def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit

def get_joint_limits(body, joint):
    # TODO: make a version for several joints?
    if is_circular(body, joint):
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit

def get_min_limit(body, joint):
    return get_joint_limits(body, joint)[0]

def get_max_limit(body, joint):
    return get_joint_limits(body, joint)[1]

def get_max_velocity(body, joint):
    return get_joint_info(body, joint).jointMaxVelocity

def get_max_force(body, joint):
    return get_joint_info(body, joint).jointMaxForce

def get_joint_q_index(body, joint):
    return get_joint_info(body, joint).qIndex

def get_joint_v_index(body, joint):
    return get_joint_info(body, joint).uIndex

def get_joint_axis(body, joint):
    return get_joint_info(body, joint).jointAxis

def get_joint_parent_frame(body, joint):
    joint_info = get_joint_info(body, joint)
    return joint_info.parentFramePos, joint_info.parentFrameOrn

def violates_limit(body, joint, value):
    if not is_circular(body, joint):
        lower, upper = get_joint_limits(body, joint)
        if (value < lower) or (upper < value):
            return True
    return False

def violates_limits(body, joints, values):
    return any(violates_limit(body, joint, value) for joint, value in zip(joints, values))

def wrap_joint(body, joint, value):
    if is_circular(body, joint):
        return wrap_angle(value)
    return value

#####################################

# Links

BASE_LINK = -1
STATIC_MASS = 0

get_num_links = get_num_joints
get_links = get_joints

def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).linkName.decode('UTF-8')

def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex

def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in get_joints(body):
        if get_link_name(body, link) == name:
            return link
    raise ValueError(body, name)


def has_link(body, name):
    try:
        link_from_name(body, name)
    except ValueError:
        return False
    return True

LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])

def get_link_state(body, link):
    return LinkState(*p.getLinkState(body, link, physicsClientId=CLIENT))

def get_com_pose(body, link): # COM = center of mass
    link_state = get_link_state(body, link)
    return link_state.linkWorldPosition, link_state.linkWorldOrientation

def get_link_inertial_pose(body, link):
    link_state = get_link_state(body, link)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation

def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_pose(body)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation

def get_all_link_parents(body):
    return {link: get_link_parent(body, link) for link in get_links(body)}

def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children

def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])

def get_link_ancestors(body, link):
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]

def get_joint_ancestors(body, link):
    return get_link_ancestors(body, link) + [link]

def get_movable_joint_ancestors(body, link):
    return list(filter(lambda j: is_movable(body, j), get_joint_ancestors(body, link)))

def get_link_descendants(body, link):
    descendants = []
    for child in get_link_children(body, link):
        descendants.append(child)
        descendants += get_link_descendants(body, child)
    return descendants

def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)

def get_adjacent_links(body):
    adjacent = set()
    for link in get_links(body):
        parent = get_link_parent(body, link)
        adjacent.add((link, parent))
        #adjacent.add((parent, link))
    return adjacent

def get_adjacent_fixed_links(body):
    return list(filter(lambda item: not is_movable(body, item[0]),
                  get_adjacent_links(body)))


def get_fixed_links(body):
    edges = defaultdict(list)
    for link, parent in get_adjacent_fixed_links(body):
        edges[link].append(parent)
        edges[parent].append(link)
    visited = set()
    fixed = set()
    for initial_link in get_links(body):
        if initial_link in visited:
            continue
        cluster = [initial_link]
        queue = deque([initial_link])
        visited.add(initial_link)
        while queue:
            for next_link in edges[queue.popleft()]:
                if next_link not in visited:
                    cluster.append(next_link)
                    queue.append(next_link)
                    visited.add(next_link)
        fixed.update(product(cluster, cluster))
    return fixed

DynamicsInfo = namedtuple('DynamicsInfo', ['mass', 'lateral_friction',
                                           'local_inertia_diagonal', 'local_inertial_pos',  'local_inertial_orn',
                                           'restitution', 'rolling_friction', 'spinning_friction',
                                           'contact_damping', 'contact_stiffness'])

def get_dynamics_info(body, link=BASE_LINK):
    return DynamicsInfo(*p.getDynamicsInfo(body, link, physicsClientId=CLIENT))

def get_mass(body, link=BASE_LINK):
    return get_dynamics_info(body, link).mass

def get_joint_inertial_pose(body, joint):
    dynamics_info = get_dynamics_info(body, joint)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn

def get_local_link_pose(body, joint):
    parent_joint = get_link_parent(body, joint)

    #world_child = get_link_pose(body, joint)
    #world_parent = get_link_pose(body, parent_joint)
    ##return multiply(invert(world_parent), world_child)
    #return multiply(world_child, invert(world_parent))

    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169
    parent_com = get_joint_parent_frame(body, joint)
    tmp_pose = invert(multiply(get_joint_inertial_pose(body, joint), parent_com))
    parent_inertia = get_joint_inertial_pose(body, parent_joint)
    #return multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
    _, orn = multiply(parent_inertia, tmp_pose)
    pos, _ = multiply(parent_inertia, Pose(parent_com[0]))
    return (pos, orn)

#####################################

# Shapes

SHAPE_TYPES = {
    p.GEOM_SPHERE: 'sphere', # 2
    p.GEOM_BOX: 'box', # 3
    p.GEOM_CYLINDER: 'cylinder', # 4
    p.GEOM_MESH: 'mesh', # 5
    p.GEOM_PLANE: 'plane',  # 6
    p.GEOM_CAPSULE: 'capsule',  # 7
}

# TODO: clean this up to avoid repeated work

def create_box(w, l, h, mass=STATIC_MASS, color=(1, 0, 0, 1)):
    half_extents = [w/2., l/2., h/2.]
    collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=CLIENT)
    if (color is None) or not has_gui():
        visual_id = -1
    else:
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=CLIENT)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT) # basePosition | baseOrientation
    # linkCollisionShapeIndices | linkVisualShapeIndices

def create_cylinder(radius, height, mass=STATIC_MASS, color=(0, 0, 1, 1)):
    collision_id =  p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=CLIENT)
    if (color is None) or not has_gui():
        visual_id = -1
    else:
        visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, height=height, rgbaColor=color, physicsClientId=CLIENT)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT) # basePosition | baseOrientation

def create_capsule(radius, height, mass=STATIC_MASS, color=(0, 0, 1, 1)):
    collision_id = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height, physicsClientId=CLIENT)
    if (color is None) or not has_gui():
        visual_id = -1
    else:
        visual_id = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, height=height, rgbaColor=color, physicsClientId=CLIENT)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT) # basePosition | baseOrientation

def create_sphere(radius, mass=0, color=(0,0,1,1), pos = None):
    urdf_r = 0.01
    scale_factor = radius/urdf_r 
    if color == (1,0,0,1):
        filename = 'urdf/sphere/red_sphere.urdf'
    else: 
        filename = 'urdf/sphere/blue_sphere.urdf'
    if pos is None:
        return p.loadURDF(path+filename, globalScaling=scale_factor)
    else:
        return p.loadURDF(path+filename, globalScaling=scale_factor, basePosition = pos)

def create_marker(radius, mass=STATIC_MASS, color=(0, 0, 1, 1), point = (0,0,0)):
    # mass = 0  => static
    if (color is None) or not has_gui():
        visual_id = -1
    else:
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=CLIENT)
    body=  p.createMultiBody(baseMass=mass, baseVisualShapeIndex=visual_id, physicsClientId=CLIENT) # basePosition | baseOrientation
    set_point(body, point)
    return body

def create_plane(normal=[0, 0, 1], mass=STATIC_MASS, color=(.5, .5, .5, 1)):
    collision_id = p.createCollisionShape(p.GEOM_PLANE, normal=normal, physicsClientId=CLIENT)
    if (color is None) or not has_gui():
        visual_id = -1
    else:
        visual_id = p.createVisualShape(p.GEOM_PLANE, normal=normal, rgbaColor=color, physicsClientId=CLIENT)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT) # basePosition | baseOrientation

mesh_count = count()
MESH_DIR = 'temp/'

def create_mesh(mesh, scale=1, mass=STATIC_MASS, color=(.5, .5, .5, 1)):
    # http://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
    # TODO: read OFF / WRL / OBJ files
    # TODO: maintain dict to file
    ensure_dir(MESH_DIR)
    path = os.path.join(MESH_DIR, 'mesh{}.obj'.format(next(mesh_count)))
    write(path, obj_file_from_mesh(mesh))
    mesh_scale = scale*np.ones(3)
    collision_id = p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=mesh_scale, physicsClientId=CLIENT)
    if (color is None) or not has_gui():
        visual_id = -1
    else:
        visual_id = p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=mesh_scale, rgbaColor=color, physicsClientId=CLIENT)
    #safe_remove(path) # TODO: removing might delete mesh?
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT) # basePosition | baseOrientation


VisualShapeData = namedtuple('VisualShapeData', ['objectUniqueId', 'linkIndex',
                                                 'visualGeometryType', 'dimensions', 'meshAssetFileName',
                                                 'localVisualFrame_position', 'localVisualFrame_orientation',
                                                 'rgbaColor'])

def visual_shape_from_data(data, client):
    if (data.visualGeometryType == p.GEOM_MESH) and (data.meshAssetFileName == 'unknown_file'):
        return -1
    # visualFramePosition: translational offset of the visual shape with respect to the link
    # visualFrameOrientation: rotational offset (quaternion x,y,z,w) of the visual shape with respect to the link frame
    pose = (data.localVisualFrame_position, data.localVisualFrame_orientation)
    inertial_pose = get_joint_inertial_pose(data.objectUniqueId, data.linkIndex)
    point, quat = multiply(invert(inertial_pose), pose)
    return p.createVisualShape(shapeType=data.visualGeometryType,
                               radius=get_data_radius(data),
                               halfExtents=np.array(get_data_extents(data))/2,
                               length=get_data_height(data),
                               fileName=data.meshAssetFileName,
                               meshScale=get_data_scale(data),
                               planeNormal=get_data_normal(data),
                               rgbaColor=data.rgbaColor,
                               #specularColor=,
                               visualFramePosition=point,
                               visualFrameOrientation=quat,
                               physicsClientId=client)

def get_visual_data(body, link=BASE_LINK):
    visual_data = [VisualShapeData(*tup) for tup in p.getVisualShapeData(body, physicsClientId=CLIENT)]
    return list(filter(lambda d: d.linkIndex == link, visual_data))

# object_unique_id and linkIndex seem to be noise
CollisionShapeData = namedtuple('CollisionShapeData', ['object_unique_id', 'linkIndex',
                                                       'geometry_type', 'dimensions', 'filename',
                                                       'local_frame_pos', 'local_frame_orn'])

def collision_shape_from_data(data, body, link, client):
    if (data.geometry_type == p.GEOM_MESH) and (data.filename == 'unknown_file'):
        return -1
    pose = (data.local_frame_pos, data.local_frame_orn)
    pose = multiply(invert(get_joint_inertial_pose(body, link)), pose)
    point, quat = pose
    # TODO: the visual data seems affected by the collision data
    return p.createCollisionShape(shapeType=data.geometry_type,
                                  radius=get_data_radius(data),
                                  # halfExtents=get_data_extents(data.geometry_type, data.dimensions),
                                  halfExtents=np.array(get_data_extents(data)) / 2,
                                  height=get_data_height(data),
                                  fileName=data.filename.decode(encoding='UTF-8'),
                                  meshScale=get_data_scale(data),
                                  planeNormal=get_data_normal(data),
                                  collisionFramePosition=point,
                                  collisionFrameOrientation=quat,
                                  physicsClientId=client)
    #return p.createCollisionShapeArray()

def clone_visual_shape(body, link, client):
    if not has_gui(client):
        return -1
    visual_data = get_visual_data(body, link)
    if visual_data:
        assert (len(visual_data) == 1)
        return visual_shape_from_data(visual_data[0], client)
    return -1

def clone_collision_shape(body, link, client):
    collision_data = get_collision_data(body, link)
    if collision_data:
        assert (len(collision_data) == 1)
        # TODO: can do CollisionArray
        return collision_shape_from_data(collision_data[0], body, link, client)
    return -1

def clone_body(body, links=None, collision=True, visual=True, client=None):
    # TODO: names are not retained
    # TODO: error with createMultiBody link poses on PR2
    # localVisualFrame_position: position of local visual frame, relative to link/joint frame
    # localVisualFrame orientation: orientation of local visual frame relative to link/joint frame
    # parentFramePos: joint position in parent frame
    # parentFrameOrn: joint orientation in parent frame
    client = get_client(client)
    if links is None:
        links = get_links(body)
    #movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = get_link_parent(body, links[0]) if links else BASE_LINK
    new_from_original[base_link] = -1

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = [] # list of local link positions, with respect to parent
    orientations = [] # list of local link orientations, w.r.t. parent
    inertial_positions = [] # list of local inertial frame pos. in link frame
    inertial_orientations = [] # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link)
        dynamics_info = get_dynamics_info(body, link)
        masses.append(dynamics_info.mass)
        collision_shapes.append(clone_collision_shape(body, link, client) if collision else -1)
        visual_shapes.append(clone_visual_shape(body, link, client) if visual else -1)
        point, quat = get_local_link_pose(body, link)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1) # TODO: need the increment to work
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)
    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169

    base_dynamics_info = get_dynamics_info(body, base_link)
    base_point, base_quat = get_link_pose(body, base_link)
    new_body = p.createMultiBody(baseMass=base_dynamics_info.mass,
                                 baseCollisionShapeIndex=clone_collision_shape(body, base_link, client) if collision else -1,
                                 baseVisualShapeIndex=clone_visual_shape(body, base_link, client) if visual else -1,
                                 basePosition=base_point,
                                 baseOrientation=base_quat,
                                 baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
                                 baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
                                 linkMasses=masses,
                                 linkCollisionShapeIndices=collision_shapes,
                                 linkVisualShapeIndices=visual_shapes,
                                 linkPositions=positions,
                                 linkOrientations=orientations,
                                 linkInertialFramePositions=inertial_positions,
                                 linkInertialFrameOrientations=inertial_orientations,
                                 linkParentIndices=parent_indices,
                                 linkJointTypes=joint_types,
                                 linkJointAxis=joint_axes,
                                 physicsClientId=client)
    #set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    for joint, value in zip(range(len(links)), get_joint_positions(body, links)):
        # TODO: check if movable?
        p.resetJointState(new_body, joint, value, physicsClientId=client)
    return new_body

def clone_world(client=None, exclude=[]):
    mapping = {}
    for body in get_bodies():
        if body not in exclude:
            new_body = clone_body(body, collision=True, visual=True, client=client)
            mapping[body] = new_body
    return mapping

#####################################

def get_collision_data(body, link=BASE_LINK):
    return [CollisionShapeData(*tup) for tup in p.getCollisionShapeData(body, link, physicsClientId=CLIENT)]

def get_data_type(data):
    return data.geometry_type if isinstance(data, CollisionShapeData) else data.visualGeometryType

def get_data_filename(data):
    return data.filename if isinstance(data, CollisionShapeData) else data.meshAssetFileName

def get_data_extents(data):
    """
    depends on geometry type:
    for GEOM_BOX: extents,
    for GEOM_SPHERE dimensions[0] = radius,
    for GEOM_CAPSULE and GEOM_CYLINDER, dimensions[0] = height (length), dimensions[1] = radius.
    For GEOM_MESH, dimensions is the scaling factor.
    :return:
    """
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_BOX:
        return dimensions
    return [1, 1, 1]

def get_data_radius(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_SPHERE:
        return dimensions[0]
    if geometry_type in (p.GEOM_SPHERE, p.GEOM_CAPSULE):
        return dimensions[1]
    return 0.5

def get_data_height(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type in (p.GEOM_SPHERE, p.GEOM_CAPSULE):
        return dimensions[0]
    return 1

def get_data_scale(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_MESH:
        return dimensions
    return [1, 1, 1]

def get_data_normal(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_PLANE:
        return dimensions
    return [0, 0, 1]

def get_data_geometry(data):
    geometry_type = get_data_type(data)
    if geometry_type == p.GEOM_SPHERE:
        parameters = [get_data_radius(data)]
    elif geometry_type == p.GEOM_BOX:
        parameters = [get_data_extents(data)]
    elif geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        parameters = [get_data_height(data), get_data_radius(data)]
    elif geometry_type == p.GEOM_MESH:
        parameters = [get_data_filename(data), get_data_scale(data)]
    elif geometry_type == p.GEOM_PLANE:
        parameters = [get_data_extents(data)]
    else:
        raise ValueError(geometry_type)
    return SHAPE_TYPES[geometry_type], parameters

def set_color(body, color, link=BASE_LINK, shape_index=-1):
    """
    Experimental for internal use, recommended ignore shapeIndex or leave it -1.
    Intention was to let you pick a specific shape index to modify,
    since URDF (and SDF etc) can have more than 1 visual shape per link.
    This shapeIndex matches the list ordering returned by getVisualShapeData.
    :param body:
    :param link:
    :param shape_index:
    :return:
    """
    return p.changeVisualShape(body, link, rgbaColor=color, physicsClientId=CLIENT)

#####################################

# Bounding box

def get_lower_upper(body, link=BASE_LINK):
    # TODO: only gets AABB for a single link
    return p.getAABB(body, linkIndex=link, physicsClientId=CLIENT)

get_aabb = get_lower_upper

def get_center_extent(body):
    lower, upper = get_aabb(body)
    center = (np.array(lower) + upper) / 2
    extents = (np.array(upper) - lower)
    return center, extents

def aabb2d_from_aabb(aabb):
    (lower, upper) = aabb
    return lower[:2], upper[:2]

def aabb_contains_aabb(contained, container):
    lower1, upper1 = contained
    lower2, upper2 = container
    return np.all(lower2 <= lower1) and np.all(upper1 <= upper2)

def aabb_contains_point(point, container):
    lower, upper = container
    return np.all(lower <= point) and np.all(point <= upper)

def get_bodies_in_region(aabb):
    (lower, upper) = aabb
    return p.getOverlappingObjects(lower, upper, physicsClientId=CLIENT)

#####################################

# Collision

#MAX_DISTANCE = 1e-3
MAX_DISTANCE = 0

def contact_collision():
    step_simulation()
    return len(p.getContactPoints(physicsClientId=CLIENT)) != 0

ContactResult = namedtuple('ContactResult', ['contactFlag', 'bodyUniqueIdA', 'bodyUniqueIdB',
                                         'linkIndexA', 'linkIndexB', 'positionOnA', 'positionOnB',
                                         'contactNormalOnB', 'contactDistance', 'normalForce'])

def pairwise_collision(body1, body2, max_distance=MAX_DISTANCE): # 10000
    # TODO: confirm that this doesn't just check the base link
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  physicsClientId=CLIENT)) != 0 # getContactPoints


def pairwise_link_collision(body1, link1, body2, link2, max_distance=MAX_DISTANCE): # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2,
                                  physicsClientId=CLIENT)) != 0 # getContactPoints

#def single_collision(body, max_distance=1e-3):
#    return len(p.getClosestPoints(body, max_distance=max_distance)) != 0

def single_collision(body1, **kwargs):
    for body2 in get_bodies():
        if (body1 != body2) and pairwise_collision(body1, body2, **kwargs):
            return True
    return False

def all_collision(**kwargs):
    bodies = get_bodies()
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            if pairwise_collision(bodies[i], bodies[j], **kwargs):
                return True
    return False

RayResult = namedtuple('RayResult', ['objectUniqueId', 'linkIndex',
                                     'hit_fraction', 'hit_position', 'hit_normal'])

def ray_collision(rays):
    ray_starts = [start for start, _ in rays]
    ray_ends = [start for _, end in rays]
    return [RayResult(*tup) for tup in p.rayTestBatch(ray_starts, ray_ends, physicsClientId=CLIENT)]
    #return RayResult(*p.rayTest(start, end))

#####################################

# Joint motion planning

def get_sample_fn(body, joints):
    def fn():
        values = []
        for joint in joints:
            limits = CIRCULAR_LIMITS if is_circular(body, joint) \
                else get_joint_limits(body, joint)
            values.append(np.random.uniform(*limits))
        return tuple(values)
    return fn

def get_difference_fn(body, joints):
    def fn(q2, q1):
        difference = []
        for joint, value2, value1 in zip(joints, q2, q1):
            difference.append(circular_difference(value2, value1)
                              if is_circular(body, joint) else (value2 - value1))
        return tuple(difference)
    return fn

def get_distance_fn(body, joints, weights=None):
    # TODO: custom weights and step sizes
    if weights is None:
        weights = 1*np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn

def get_refine_fn(body, joints, num_steps=0):
    difference_fn = get_difference_fn(body, joints)
    num_steps = num_steps + 1
    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            q = tuple((1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q)
            yield q
            # TODO: should wrap these joints
    return fn

def refine_path(body, joints, waypoints, num_steps):
    refine_fn = get_refine_fn(body, joints, num_steps)
    refined_path = []
    for v1, v2 in zip(waypoints, waypoints[1:]):
        refined_path += list(refine_fn(v1, v2))
    return refined_path

def get_extend_fn(body, joints, resolutions=None):
    if resolutions is None:
        resolutions = 0.05*np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        refine_fn = get_refine_fn(body, joints, num_steps=int(np.max(steps)))
        return refine_fn(q1, q2)
    return fn

# def sparsify_path(body, joints, path):
#     if len(path) <= 2:
#         return path
#     difference_fn = get_difference_fn(body, joints)
#     waypoints = [path[0]]
#     last_difference = difference_fn(waypoints[-1], path[1])
#     last_conf = path[1]
#     for q in path[2:]:
#         new_difference = difference_fn(waypoints[-1], q)
#         #if np.allclose(last_difference, new_difference, atol=1e-3, rtol=0):
#         #
#         #last_difference = new_difference
#         #last_conf = q
#         # TODO: test if a scaling of itself
#     return path

def waypoints_from_path(path):
    if len(path) < 2:
        return path

    def difference_fn(q2, q1):
        return np.array(q2) - np.array(q1)

    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=1e-3, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints

def get_moving_links(body, moving_joints):
    moving_links = list(moving_joints)
    for link in moving_joints:
        moving_links += get_link_descendants(body, link)
    return list(set(moving_links))

def get_moving_pairs(body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    moving_links = get_moving_links(body, moving_joints)
    for i in range(len(moving_links)):
        link1 = moving_links[i]
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        for j in range(i+1, len(moving_links)):
            link2 = moving_links[j]
            ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
            if ancestors1 != ancestors2:
                yield link1, link2


def get_self_link_pairs(body, joints, disabled_collisions=set()):
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if True:
        check_link_pairs += list(get_moving_pairs(body, joints))
    else:
        check_link_pairs += list(combinations(moving_links, 2))
    check_link_pairs = list(filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs


def get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions):
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
    moving_bodies = [body] + [attachment.child for attachment in attachments]
    if obstacles is None:
        obstacles = list(set(get_bodies()) - set(moving_bodies))
    check_body_pairs = list(product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))
    # TODO: maybe prune the link adjacent to the robot

    # TODO: end-effector constraints
    def collision_fn(q):
        if violates_limits(body, joints, q):
            return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        #if pairwise_collision(body, body):
        #    return True
        for link1, link2 in check_link_pairs:
            if pairwise_link_collision(body, link1, body, link2):
                return True
        return any(pairwise_collision(*pair) for pair in check_body_pairs)
    return collision_fn

def check_initial_end(start_conf, end_conf, collision_fn):
    if collision_fn(start_conf):
        print("Warning: initial configuration is in collision")
        return False
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return False
    return True

def plan_waypoints_joint_motion(body, joints, waypoints, obstacles=None, attachments=[],
                      self_collisions=True, disabled_collisions=set()):
    extend_fn = get_extend_fn(body, joints)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions)
    start_conf = get_joint_positions(body, joints)
    if not check_initial_end(start_conf, ([start_conf] + waypoints)[-1], collision_fn):
        return None
    path = [start_conf]
    for waypoint in waypoints:
        assert len(joints) == len(waypoint)
        for q in extend_fn(path[-1], waypoint):
            if collision_fn(q):
                return None
            path.append(q)
    return path

def plan_direct_joint_motion(body, joints, end_conf, obstacles=None, attachments=[],
                      self_collisions=True, disabled_collisions=set()):
    return plan_waypoints_joint_motion(body, joints, [end_conf], obstacles, attachments, self_collisions,
                                       disabled_collisions)

def plan_joint_motion(body, joints, end_conf, obstacles=None, attachments=[],
                      self_collisions=True, disabled_collisions=set(), direct=False, **kwargs):
    if direct:
        return plan_direct_joint_motion(body, joints, end_conf, obstacles, attachments, self_collisions, disabled_collisions)
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints)
    distance_fn = get_distance_fn(body, joints)
    extend_fn = get_extend_fn(body, joints)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions)
    # TODO: test self collision with the holding

    start_conf = get_joint_positions(body, joints)
    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    #if direct:
    #    return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)

#####################################

# Pose motion planning

def plan_base_motion(body, end_conf, base_limits, obstacles=None, direct=False,
                     weights=1*np.ones(3), resolutions=0.05*np.ones(3), **kwargs):
    def sample_fn():
        x, y = np.random.uniform(*base_limits)
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    def difference_fn(q2, q1):
        dx, dy = np.array(q2[:2]) - np.array(q1[:2])
        dtheta = circular_difference(q2[2], q1[2])
        return (dx, dy, dtheta)

    def distance_fn(q1, q2):
        difference = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, difference * difference))

    def extend_fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n = int(np.max(steps)) + 1
        q = q1
        for i in range(n):
            q = tuple((1. / (n - i)) * np.array(difference_fn(q2, q)) + q)
            yield q
            # TODO: should wrap these joints

    def collision_fn(q):
        # TODO: update this function
        set_base_values(body, q)
        if obstacles is None:
            return single_collision(body)
        return any(pairwise_collision(body, obs) for obs in obstacles)

    start_conf = get_base_values(body)
    if collision_fn(start_conf):
        print("Warning: initial configuration is in collision")
        return None
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn,
                 sample_fn, extend_fn, collision_fn, **kwargs)

#####################################

# Placements

def stable_z(body, surface):
    _, extent = get_center_extent(body)
    _, upper = get_lower_upper(surface)
    return (upper + extent/2)[2]

def is_placement(body, surface, epsilon=1e-2): # TODO: above / below
    top_aabb = get_lower_upper(body)
    bottom_aabb = get_lower_upper(surface)
    bottom_z_max = bottom_aabb[1][2]
    return (bottom_z_max <= top_aabb[0][2] <= (bottom_z_max + epsilon)) and \
           (aabb_contains_aabb(aabb2d_from_aabb(top_aabb), aabb2d_from_aabb(bottom_aabb)))

def is_center_stable(body, surface, epsilon=1e-2):
    # TODO: compute AABB in origin
    # TODO: use center of mass?
    center, extent = get_center_extent(body)
    base_center = center - np.array([0, 0, extent[2]])/2
    bottom_aabb = get_aabb(surface)
    bottom_z_max = bottom_aabb[1][2]
    #return (bottom_z_max <= base_center[2] <= (bottom_z_max + epsilon)) and \
    return (abs(base_center[2] - bottom_z_max) < epsilon) and \
           (aabb_contains_point(base_center[:2], aabb2d_from_aabb(bottom_aabb)))


def sample_placement(top_body, bottom_body, max_attempts=50, epsilon=1e-3):
    bottom_aabb = get_lower_upper(bottom_body)
    for _ in range(max_attempts):
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        quat = z_rotation(theta)
        set_quat(top_body, quat)
        center, extent = get_center_extent(top_body)
        lower = (np.array(bottom_aabb[0]) + extent/2)[:2]
        upper = (np.array(bottom_aabb[1]) - extent/2)[:2]
        if np.any(upper < lower):
          continue
        x, y = np.random.uniform(lower, upper)
        z = (bottom_aabb[1] + extent/2.)[2] + epsilon
        point = np.array([x, y, z]) + (get_point(top_body) - center)
        set_point(top_body, point)
        return point, quat
    return None

#####################################

# Reachability

def sample_reachable_base(robot, point, reachable_range=(0.25, 1.0), max_attempts=50):
    for _ in range(max_attempts):
        radius = np.random.uniform(*reachable_range)
        x, y = radius*unit_from_theta(np.random.uniform(-np.pi, np.pi)) + point[:2]
        yaw = np.random.uniform(*CIRCULAR_LIMITS)
        base_values = (x, y, yaw)
        #set_base_values(robot, base_values)
        return base_values
    return None

def uniform_pose_generator(robot, gripper_pose, **kwargs):
    point = point_from_pose(gripper_pose)
    while True:
        base_values = sample_reachable_base(robot, point)
        if base_values is None:
            break
        yield base_values
        #set_base_values(robot, base_values)
        #yield get_pose(robot)

#####################################

# Constraints - applies forces when not satisfied

def get_constraints():
    """
    getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
    Note that the constraint unique ids may not be contiguous, since you may remove constraints.
    """
    return [p.getConstraintUniqueId(i, physicsClientId=CLIENT)
            for i in range(p.getNumConstraints(physicsClientId=CLIENT))]

def remove_constraint(constraint):
    p.removeConstraint(constraint, physicsClientId=CLIENT)

ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild', 'maxAppliedForce'])

def get_constraint_info(constraint): # getConstraintState
    # TODO: four additional arguments
    return ConstraintInfo(*p.getConstraintInfo(constraint, physicsClientId=CLIENT)[:11])

def get_fixed_constraints():
    fixed_constraints = []
    for constraint in get_constraints():
        constraint_info = get_constraint_info(constraint)
        if constraint_info.constraintType == p.JOINT_FIXED:
            fixed_constraints.append(constraint)
    return fixed_constraints

def add_fixed_constraint(body, robot, robot_link, max_force=None):
    body_link = BASE_LINK
    body_pose = get_pose(body)
    end_effector_pose = get_link_pose(robot, robot_link)
    grasp_pose = multiply(invert(end_effector_pose), body_pose)
    point, quat = grasp_pose
    # TODO: can I do this when I'm not adjacent?
    # joint axis in local frame (ignored for JOINT_FIXED)
    #return p.createConstraint(robot, robot_link, body, body_link,
    #                          p.JOINT_FIXED, jointAxis=unit_point(),
    #                          parentFramePosition=unit_point(),
    #                          childFramePosition=point,
    #                          parentFrameOrientation=unit_quat(),
    #                          childFrameOrientation=quat)
    constraint = p.createConstraint(robot, robot_link, body, body_link,  # Both seem to work
                              p.JOINT_FIXED, jointAxis=unit_point(),
                                    parentFramePosition=point,
                                    childFramePosition=unit_point(),
                                    parentFrameOrientation=quat,
                                    childFrameOrientation=unit_quat(),
                                    physicsClientId=CLIENT)
    if max_force is not None:
        p.changeConstraint(constraint, maxForce=max_force, physicsClientId=CLIENT)
    return constraint

def remove_fixed_constraint(body, robot, robot_link):
    for constraint in get_fixed_constraints():
        constraint_info = get_constraint_info(constraint)
        if (body == constraint_info.childBodyUniqueId) and \
                (BASE_LINK == constraint_info.childLinkIndex) and \
                (robot == constraint_info.parentBodyUniqueId) and \
                (robot_link == constraint_info.parentJointIndex):
            remove_constraint(constraint)

#####################################

# Grasps

GraspInfo = namedtuple('GraspInfo', ['get_grasps', 'approach_pose'])

class Attachment(object):
    def __init__(self, parent, parent_link, grasp_pose, child):
        self.parent = parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child
        #self.child_link = child_link # child_link=BASE_LINK
    def assign(self):
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose)
        return child_pose
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.parent, self.child)

def body_from_end_effector(end_effector_pose, grasp_pose):
    """
    world_from_parent * parent_from_child = world_from_child
    """
    return multiply(end_effector_pose, grasp_pose)

def end_effector_from_body(body_pose, grasp_pose):
    """
    world_from_child * (parent_from_child)^(-1) = world_from_parent
    """
    return multiply(body_pose, invert(grasp_pose))

def approach_from_grasp(approach_pose, end_effector_pose):
    return multiply(approach_pose, end_effector_pose)

def get_grasp_pose(constraint):
    """
    Grasps are parent_from_child
    """
    constraint_info = get_constraint_info(constraint)
    assert(constraint_info.constraintType == p.JOINT_FIXED)
    joint_from_parent = (constraint_info.jointPivotInParent, constraint_info.jointFrameOrientationParent)
    joint_from_child = (constraint_info.jointPivotInChild, constraint_info.jointFrameOrientationChild)
    return multiply(invert(joint_from_parent), joint_from_child)

#####################################

# Control

def control_joint(body, joint, value):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=0.4*get_max_force(body, joint),
                                   physicsClientId=CLIENT)

def weaken(body):
    num_joints = p.getNumJoints(body)
    for i in range(num_joints):
        p.setJointMotorControl2(bodyIndex=body,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=0,force=0,positionGain=0.3,velocityGain=1, targetVelocity=0)


def control_joints(body, joints, positions, heavy_joints=[], heavy_confs=[], force_scale = 1200):
    # TODO: the whole PR2 seems to jitter
    #kp = 1.0
    #kv = 0.3
    """
    return p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       targetVelocities=[0.0] * len(joints),
                                       physicsClientId=CLIENT) #,
                                        #positionGains=[kp] * len(joints),
                                        #velocityGains=[kv] * len(joints),)
                                        #forces=forces)
    """
    for i in range(len(joints)):
        p.setJointMotorControl2(bodyIndex=body,jointIndex=joints[i],controlMode=p.POSITION_CONTROL,targetPosition=positions[i],force=force_scale, targetVelocity=0)
    """
    for i in range(len(heavy_joints)):
        p.setJointMotorControl2(bodyIndex=body,jointIndex=heavy_joints[i],controlMode=p.POSITION_CONTROL,targetPosition=heavy_confs[i],force=5000,positionGain=0.3,velocityGain=1, targetVelocity=0)
    
    """
    simulate_for_duration(0.1)
    return 


def joint_controller(body, joints, target, max_time=None, heavy_joints=[], heavy_confs=[], force = 300):
    assert(len(joints) == len(target))
    iteration = 0
    while not np.allclose(get_joint_positions(body, joints), target, atol=1e-3, rtol=0):
        control_joints(body, joints, target, heavy_joints=heavy_joints, heavy_confs=heavy_confs, force_scale=force)
        yield iteration
        iteration += 1

def joint_controller_hold(body, joints, target, max_time=None):
    """
    Keeps other joints in place
    """
    movable_joints = get_movable_joints(body)
    movable_from_original = {o: m for m, o in enumerate(movable_joints)}
    conf = list(get_joint_positions(body, movable_joints))
    for joint, value in zip(joints, target):
        conf[movable_from_original[joint]] = value
    return joint_controller(body, movable_joints, conf)

def velocity_control_joints(body, joints, velocities):
    #kv = 0.3
    return p.setJointMotorControlArray(body, joints, p.VELOCITY_CONTROL,
                                       targetVelocities=velocities,
                                       physicsClientId=CLIENT) #,
                                        #velocityGains=[kv] * len(joints),)
                                        #forces=forces)

#####################################

def inverse_kinematics(robot, link, pose, max_iterations=200, tolerance=1e-3):
    (target_point, target_quat) = pose
    movable_joints = get_movable_joints(robot)
    for iterations in range(max_iterations):
        # TODO: stop is no progress
        # TODO: stop if collision or invalid joint limits
        kinematic_conf = p.calculateInverseKinematics(robot, link, target_point, target_quat,
                                                      physicsClientId=CLIENT)
        if (kinematic_conf is None) or any(map(math.isnan, kinematic_conf)):
            return None
        set_joint_positions(robot, movable_joints, kinematic_conf)
        link_point, link_quat = get_link_pose(robot, link)
        if np.allclose(link_point, target_point, atol=tolerance, rtol=0) and \
                np.allclose(link_quat, target_quat, atol=tolerance, rtol=0):
            break
    else:
        return None
    if violates_limits(robot, movable_joints, kinematic_conf):
        return None
    return kinematic_conf

def workspace_trajectory(robot, link, start_point, direction, quat, step_size=0.01, **kwargs):
    # TODO: pushing example
    # TODO: just use current configuration?
    # TODO: check collisions?
    # TODO: lower intermediate tolerance
    distance = get_length(direction)
    unit_direction = get_unit_vector(direction)
    traj = []
    for t in np.arange(0, distance, step_size):
        point = start_point + t*unit_direction
        pose = (point, quat)
        conf = inverse_kinematics(robot, link, pose, **kwargs)
        if conf is None:
            return None
        traj.append(conf)
    return traj

#####################################

def call_ik_fast(robot, target_pose, target_quat):
    target_quat = list(target_quat)
    # this code has quaternions in [x,y,z,w] for some reason
    # the solver uses [w, x, y, z] like everyone else
    target_quat = [target_quat[3]] + target_quat[:3]
    torso_joint = joint_from_name(robot, 'torso_lift_joint')
    torso = get_joint_position(robot, torso_joint)
    # This is needed because the .dae file the C++ IK was generated with didn't include the force sensor extension
    adjustment = [-0.0356, 0, 0] # add -0.0356 in whatever the orientation is
    adjustment = apply_transformation([adjustment, [1, 0, 0, 0]], quat=target_quat)
    adjustment = adjustment[0]
    target_pose = apply_transformation([target_pose, target_quat], pos=adjustment)
    target_pos = target_pose[0]
    target_quat = target_pose[1]

    # print(arm, target_pos, target_quat, torso)
    sol = arm_ik('r', target_pos, target_quat, torso)
    return sol

def sub_inverse_kinematics(robot, first_joint, target_link, target_pose, selected_links_input=None, max_iterations=2, tolerance=1e-3):
    # TODO: fix stationary joints
    # TODO: pass in set of movable joints and take least common ancestor
    # TODO: update with most recent bullet updates
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics_husky_kuka.py
    if selected_links_input is None:
        selected_links = [first_joint] + get_link_descendants(robot, first_joint)
    else:
        selected_links = selected_links_input
    selected_movable_joints = [joint for joint in selected_links if is_movable(robot, joint)]
    assert(target_link in selected_links)
    selected_target_link = selected_links.index(target_link)
    sub_robot = clone_body(robot, links=selected_links, visual=False, collision=False) # TODO: joint limits
    ll = [get_joint_limits(sub_robot, j)[0] for j in range(p.getNumJoints(sub_robot))]
    ul = [get_joint_limits(sub_robot, j)[1] for j in range(p.getNumJoints(sub_robot))]
    jr = np.subtract(ul, ll)

    (target_point, target_quat) = target_pose
    sub_movable_joints = get_movable_joints(sub_robot)
    current_conf =  get_configuration(sub_robot)
    for _ in range(max_iterations):
        sub_kinematic_conf = p.calculateInverseKinematics(sub_robot, selected_target_link,
                                                          target_point, target_quat,
                                                          physicsClientId=CLIENT, 
                                                          #restPoses = current_conf,
                                                          #jointRanges = jr,
                                                          residualThreshold=tolerance,
                                                          maxNumIterations=max_iterations,
          
                                                          lowerLimits=ll, upperLimits=ul)
        """
        if (sub_kinematic_conf is None) or any(map(math.isnan, sub_kinematic_conf)):
            print("does if statement")
            remove_body(sub_robot)
            return None
        """
	#set_joint_positions(sub_robot, sub_movable_joints, sub_kinematic_conf)
        set_joint_positions(sub_robot, sub_movable_joints, sub_kinematic_conf)
        
        link_point, link_quat = get_link_pose(sub_robot, selected_target_link)
        # TODO: let target_quat be None
        if np.allclose(link_point, target_point, atol=tolerance, rtol=0) and \
                np.allclose(link_quat, target_quat, atol=tolerance, rtol=0):
            break
        else:
            continue

    #else:
    #    remove_body(sub_robot)
    remove_body(sub_robot)
    
    joint_controller(robot, selected_movable_joints, sub_kinematic_conf, max_time=100)
    kinematic_conf = get_configuration(robot)
    if violates_limits(robot, get_movable_joints(robot), kinematic_conf):
        pass
    #return kinematic_conf
    return sub_kinematic_conf, selected_movable_joints

#####################################

def get_lifetime(lifetime):
    if lifetime is None:
        return 0
    return lifetime

def add_text(text, position=(0, 0, 0), color=(0, 0, 0), lifetime=None, parent=-1, parent_link=BASE_LINK):
    return p.addUserDebugText(text, textPosition=position, textColorRGB=color, # textSize=1,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)

def add_line(start, end, color=(0, 0, 0), width=1, lifetime=None, parent=-1, parent_link=BASE_LINK):
    return p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=width,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)

def remove_debug(debug): # removeAllUserDebugItems
    p.removeUserDebugItem(debug, physicsClientId=CLIENT)

def add_body_name(body, **kwargs):
    with PoseSaver(body):
        set_pose(body, unit_pose())
        lower, upper = get_aabb(body)
    #position = (0, 0, upper[2])
    position = upper
    return add_text(get_name(body), position=position, parent=body, **kwargs)  # removeUserDebugItem


def add_segments(points, closed=False, **kwargs):
    lines = []
    for v1, v2 in zip(points, points[1:]):
        lines.append(add_line(v1, v2, **kwargs))
    if closed:
        lines.append(add_line(points[-1], points[0], **kwargs))
    return lines


def draw_base_limits(limits, z=1e-2, **kwargs):
    lower, upper = limits
    vertices = [(lower[0], lower[1], z), (lower[0], upper[1], z),
                (upper[0], upper[1], z), (upper[0], lower[1], z)]
    return add_segments(vertices, closed=True, **kwargs)

#####################################

# Polygonal surfaces

def create_rectangular_surface(width, length):
    extents = np.array([width, length, 0]) / 2.
    unit_corners = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
    return [np.append(c, 0) * extents for c in unit_corners]

def is_point_in_polygon(point, polygon):
    sign = None
    for i in range(len(polygon)):
        v1, v2 = polygon[i - 1][:2], polygon[i][:2]
        delta = v2 - v1
        normal = np.array([-delta[1], delta[0]])
        dist = normal.dot(point[:2] - v1)
        if i == 0:  # TODO: equality?
            sign = np.sign(dist)
        elif np.sign(dist) != sign:
            return False
    return True

def apply_affine(affine, points):
    # TODO: version which applies to one point
    return [point_from_pose(multiply(affine, Pose(point=p))) for p in points]

def is_mesh_on_surface(polygon, world_from_surface, mesh, world_from_mesh, epsilon=1e-2):
    surface_from_mesh = multiply(invert(world_from_surface), world_from_mesh)
    points_surface = apply_affine(surface_from_mesh, mesh.vertices)
    min_z = np.min(points_surface[:, 2])
    return (abs(min_z) < epsilon) and \
           all(is_point_in_polygon(p, polygon) for p in points_surface)

def is_point_on_surface(polygon, world_from_surface, point_world):
    [point_surface] = apply_affine(invert(world_from_surface), [point_world])
    return is_point_in_polygon(point_surface, polygon[::-1])

def sample_polygon_tform(polygon, points):
    min_z = np.min(points[:, 2])
    aabb_min = np.min(polygon, axis=0)
    aabb_max = np.max(polygon, axis=0)
    while True:
        x = np.random.uniform(aabb_min[0], aabb_max[0])
        y = np.random.uniform(aabb_min[1], aabb_max[1])
        theta = np.random.uniform(0, 2 * np.pi)
        point = Point(x, y, -min_z)
        quat = Euler(yaw=theta)
        surface_from_origin = Pose(point, quat)
        yield surface_from_origin
        # if all(is_point_in_polygon(p, polygon) for p in apply_affine(surface_from_origin, points)):
        #  yield surface_from_origin

def sample_surface_pose(polygon, world_from_surface, mesh):
    for surface_from_origin in sample_polygon_tform(polygon, mesh.vertices):
        world_from_mesh = multiply(world_from_surface, surface_from_origin)
        if is_mesh_on_surface(polygon, world_from_surface, mesh, world_from_mesh):
            yield world_from_mesh

#####################################

# Sampling edges

def sample_categorical(categories):
    from bisect import bisect
    names = categories.keys()
    cutoffs = np.cumsum([categories[name] for name in names])/sum(categories.values())
    return names[bisect(cutoffs, np.random.random())]

def sample_edge_point(polygon, radius):
    edges = zip(polygon, polygon[-1:] + polygon[:-1])
    edge_weights = {i: max(get_length(v2 - v1) - 2 * radius, 0) for i, (v1, v2) in enumerate(edges)}
    # TODO: fail if no options
    while True:
        index = sample_categorical(edge_weights)
        v1, v2 = edges[index]
        t = np.random.uniform(radius, get_length(v2 - v1) - 2 * radius)
        yield t * get_unit_vector(v2 - v1) + v1

def get_closest_edge_point(polygon, point):
    # TODO: always pick perpendicular to the edge
    edges = zip(polygon, polygon[-1:] + polygon[:-1])
    best = None
    for v1, v2 in edges:
        proj = (v2 - v1)[:2].dot((point - v1)[:2])
        if proj <= 0:
            closest = v1
        elif get_length((v2 - v1)[:2]) <= proj:
            closest = v2
        else:
            closest = proj * get_unit_vector((v2 - v1))
        if (best is None) or (get_length((point - closest)[:2]) < get_length((point - best)[:2])):
            best = closest
    return best

def sample_edge_pose(polygon, world_from_surface, mesh):
    radius = max(get_length(v[:2]) for v in mesh.vertices)
    origin_from_base = Pose(Point(z=p.min(mesh.vertices[:, 2])))
    for point in sample_edge_point(polygon, radius):
        theta = np.random.uniform(0, 2 * np.pi)
        surface_from_origin = Pose(point, Euler(yaw=theta))
        yield multiply(world_from_surface, surface_from_origin, origin_from_base)

#####################################

# Convex Hulls

def convex_hull(points):
    # TODO: 2D convex hull
    from scipy.spatial import ConvexHull
    # TODO: cKDTree is faster, but KDTree can do all pairs closest
    hull = ConvexHull(points)
    new_indices = {i: ni for ni, i in enumerate(hull.vertices)}
    vertices = hull.points[hull.vertices, :]
    faces = np.vectorize(lambda i: new_indices[i])(hull.simplices)
    return vertices, faces

def mesh_from_points(points):
    vertices, indices = convex_hull(points)
    new_indices = []
    for triplet in indices:
        centroid = np.average(vertices[triplet], axis=0)
        v1, v2, v3 = vertices[triplet]
        normal = np.cross(v3 - v1, v2 - v1)
        if normal.dot(centroid) > 0:
            # if normal.dot(centroid) < 0:
            triplet = triplet[::-1]
        new_indices.append(tuple(triplet))
    return vertices.tolist(), new_indices

def mesh_from_body(body, link=BASE_LINK):
    # TODO: read obj files so I can always obtain the pointcloud
    # TODO: approximate cylindrical/spherical using convex hull
    # TODO: change based on geom_type
    print(get_collision_data(body, link))
    print(get_visual_data(body, link))
    # TODO: these aren't working...

#####################################

# Mesh & Pointcloud Files

def obj_file_from_mesh(mesh):
    """
    Creates a *.obj mesh string
    :param mesh: tuple of list of vertices and list of faces
    :return: *.obj mesh string
    """
    vertices, faces = mesh
    s = 'g Mesh\n' # TODO: string writer
    for v in vertices:
        assert(len(v) == 3)
        s += '\nv {}'.format(' '.join(map(str, v)))
    for f in faces:
        assert(len(f) == 3)
        f = [i+1 for i in f]
        s += '\nf {}'.format(' '.join(map(str, f)))
        s += '\nf {}'.format(' '.join(map(str, reversed(f))))
    return s

def read_mesh_off(path, scale=1.0):
    """
    Reads a *.off mesh file
    :param path: path to the *.off mesh file
    :return: tuple of list of vertices and list of faces
    """
    with open(path) as f:
        assert (f.readline().split()[0] == 'OFF'), 'Not OFF file'
        nv, nf, ne = [int(x) for x in f.readline().split()]
        verts = [tuple(scale * float(v) for v in f.readline().split()) for _ in range(nv)]
        faces = [tuple(map(int, f.readline().split()[1:])) for _ in range(nf)]
        return verts, faces


def read_pcd_file(path):
    """
    Reads a *.pcd pointcloud file
    :param path: path to the *.pcd pointcloud file
    :return: list of points
    """
    with open(path) as f:
        data = f.readline().split()
        num_points = 0
        while data[0] != 'DATA':
            if data[0] == 'POINTS':
                num_points = int(data[1])
            data = f.readline().split()
            continue
        return [tuple(map(float, f.readline().split())) for _ in range(num_points)]

# TODO: factor out things that don't depend on pybullet

#####################################

"""
def readWrl(filename, name='wrlObj', scale=1.0, color='black'):
    def readOneObj():
        vl = []
        while True:
            line = fl.readline()
            split = line.split(',')
            if len(split) != 2:
                break
            split = split[0].split()
            if len(split) == 3:
                vl.append(np.array([scale*float(x) for x in split]+[1.0]))
            else:
                break
        print '    verts', len(vl),
        verts = np.vstack(vl).T
        while line.split()[0] != 'coordIndex':
            line = fl.readline()
        line = fl.readline()
        faces = []
        while True:
            line = fl.readline()
            split = line.split(',')
            if len(split) > 3:
                faces.append(np.array([int(x) for x in split[:3]]))
            else:
                break
        print 'faces', len(faces)
        return Prim(verts, faces, hu.Pose(0,0,0,0), None,
                    name=name+str(len(prims)))

    fl = open(filename)
    assert fl.readline().split()[0] == '#VRML', 'Not VRML file?'
    prims = []
    while True:
        line = fl.readline()
        if not line: break
        split = line.split()
        if not split or split[0] != 'point':
            continue
        else:
            print 'Object', len(prims)
            prims.append(readOneObj())
    # Have one "part" so that shadows are simpler
    part = Shape(prims, None, name=name+'_part')
    # Keep color only in top entry.
    return Shape([part], None, name=name, color=color)
"""
def blacken(objID, end_index =None):
    p.changeVisualShape(objID, -1, rgbaColor=(0,0,0,1))
    if end_index is None:
        end_index =  p.getNumJoints(objID)
    for link in range(end_index):
        p.changeVisualShape(objID, link, rgbaColor=(0,0,0,1))
