## from caelan's ss-pybullet utils: https://github.com/caelan/ss-pybullet/blob/master/utils.py
from __future__ import print_function, division
import numpy as np
import pybullet as p
from motion_planners.rrt_connect import birrt, direct_path

MAX_DISTANCE = 0
PI = np.pi
CIRCULAR_LIMITS = -PI, PI

def get_pose(body):
    return p.getBasePositionAndOrientation(body)

def get_point(body):
    return p.getBasePositionAndOrientation(body)[0]

def get_bodies():
    return [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]

def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat)

def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)

def get_quat(body):
    return p.getBasePositionAndOrientation(body)[1] # [x,y,z,w]

def set_quat(body, quat):
    p.resetBasePositionAndOrientation(body, get_point(body), quat)

def get_base_values(body):
    return base_values_from_pose(get_pose(body))

def set_point(body, point):
    p.resetBasePositionAndOrientation(body, point, get_quat(body))

def z_rotation(theta):
    return quat_from_euler([0, 0, theta])

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi
    
def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)

def base_values_from_pose(pose):
    (point, quat) = pose
    x, y, _ = point
    roll, pitch, yaw = euler_from_quat(quat)
    # assert (abs(roll) < 1e-3) and (abs(pitch) < 1e-3)
    return (x, y, yaw)

def set_base_values(body, values):
    _, _, z = get_point(body)
    x, y, theta = values
    set_point(body, (x, y, z))
    set_quat(body, z_rotation(theta))

def single_collision(body1, **kwargs):
    for body2 in get_bodies():
        if (body1 != body2) and pairwise_collision(body1, body2, **kwargs):
            return True
    return False

def pairwise_collision(body1, body2, max_distance=MAX_DISTANCE): # 10000
    # TODO: confirm that this doesn't just check the base link
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)) != 0 # getContactPoints

def plan_base_motion(body, end_conf, obstacles=None, direct=False,
                     base_limits=([-2.5, -2.5], [2.5, 2.5]),
                     weights=1*np.ones(3),
                     resolutions=0.05*np.ones(3),
                     **kwargs):
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
        set_base_values(body, q)
        if obstacles is None:
            return single_collision(body)
        return any(pairwise_collision(body, obs) for obs in obstacles)

    start_conf = get_base_values(body)
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn,
                 sample_fn, extend_fn, collision_fn, **kwargs)