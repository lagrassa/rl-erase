## from caelan's ss-pybullet utils: https://github.com/caelan/ss-pybullet/blob/master/utils.py
from __future__ import print_function, division
import pybullet as p

STATIC_MASS = 0

def get_connection():
    return p.getConnectionInfo()['connectionMethod']

def has_gui():
    return get_connection() == p.GUI

def create_box(w, l, h, mass=STATIC_MASS, color=(1, 0, 0, 1)):
    half_extents = [w/2., l/2., h/2.]
    collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    if (color is None) or not has_gui():
        visual_id = -1
    else:
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id) # basePosition | baseOrientation
    # linkCollisionShapeIndices | linkVisualShapeIndices