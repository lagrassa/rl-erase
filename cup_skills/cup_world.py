from __future__ import division

import os
import pdb
import pybullet as p
import time

import numpy as np
import pybullet_data
from PIL import Image

from cup_skills.utils import create_sphere, set_point, Point, \
    get_lower_upper, simulate_for_duration, euler_from_quat
from cup_skills.local_setup import path

real_init = True
new_world = True
k = 1  # scaling factor
DEMO = False

class CupWorld:
    def __init__(self, cup_name = "cup_3.urdf", visualize=False, real_init=True, beads=True, cup_offset=(0, 0, 0), new_bead_mass=None,bead_radius=0.015, camera_distance=0.7, camera_z_offset = 0.3,
                 table=False, for_pr2=False):
        proj_dir = os.getcwd()
        self.cup_name = cup_name
        if "control" in proj_dir:
            print("Changing working directory to cup skills")
            os.chdir(proj_dir + '/../../../cup_skills')

        self.visualize = visualize
        self.camera_z_offset = camera_z_offset
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.real_init = real_init
        self.camera_distance = camera_distance
        self.num_droplets = 2
        self.for_pr3 = for_pr2
        self.radius = bead_radius

        self.table = table
        if real_init:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.simplify_viz()
        if for_pr2:
            cup_factor = 1.5
        else:
            cup_factor = 5
        self.setup(beads=True, cup_offset=cup_offset, new_bead_mass=new_bead_mass, table=table, cup_factor=cup_factor)

    def toggle_real_time(self):
        self.is_real_time = int(not self.is_real_time)
        p.setRealTimeSimulation(self.is_real_time)

    """saves where all the beads are"""
    """Returns the proportion of beads that are still in the cup"""
    def ratio_beads_in_cup(self, cup=None):
        if cup is None:
            cup = self.cupID
        aabbMin, aabbMax = p.getAABB(cup)
        all_overlapping = p.getOverlappingObjects(aabbMin, aabbMax)
        if all_overlapping is None:
            return 0
        overlapping_objects = [obj for obj in all_overlapping if obj[0] >= self.droplets[0] and obj[0] <= self.droplets[
            -1]]  # in the range to be a droplet

        if overlapping_objects is None:
            num_in = 0
        else:
            num_in = len(overlapping_objects)
        total = len(self.droplets)

        ratio = num_in / total
        assert (ratio <= 1.0)
        return ratio

    def ratio_beads_in_target(self, scoop):
        return self.ratio_beads_in_cup(scoop)



    def distance_from_cup(self, otherObj, otherLinkIndex):
        cup_pos = np.array(p.getBasePositionAndOrientation(self.cupID)[0])
        if otherLinkIndex == -1:
            other_pos = np.array(p.getBasePositionAndOrientation(otherObj)[0])

        else:
            other_pos = np.array(p.getLinkState(otherObj, otherLinkIndex)[0])
        return np.linalg.norm(cup_pos - other_pos)

    def step(self, time_step):
        if not self.is_real_time:
            simulate_for_duration(time_step, dt=0.01)
        time.sleep(0.0001)
        return

    """Fun enough, this function returns a tuple of two different camera views!"""

    def world_state(self):
        # crop to only relevant parts
        views = [0, 180]  # I have no idea why, but these seems to be in degrees
        images = ()
        for view in views:
            rgb_pixels = self.get_image_from_distance(self.cupID, self.camera_distance, z_offset=self.camera_z_offset, x_offset = 0.25, theta_offset=view)
            #from PIL import Image
            #Image.fromarray(rgb_pixels).show()
            #import ipdb; ipdb.set_trace()
            images += (rgb_pixels[:, :, 0:3],)  # decided against cropping
        return images

    def get_image_from_distance(self, obj_id, cam_distance, z_offset=0, y_offset=0, x_offset=0, theta_offset=0):
        obj_pos, obj_quat = p.getBasePositionAndOrientation(obj_id)
        adjusted_pos = (obj_pos[0] + x_offset, obj_pos[1] + y_offset, obj_pos[2] + z_offset)
        roll, pitch, yaw = euler_from_quat(obj_quat)
        yaw = yaw + theta_offset
        im_w = 128
        im_h = 128
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=adjusted_pos, distance=cam_distance,
                                                         yaw=yaw, pitch=pitch, roll=roll + np.pi, upAxisIndex=2)
        if self.visualize:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER
        fov = 60
        near_plane = 0.01
        far_plane = 500
        aspect = im_w / im_h
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

        _, _, rgb_pixels, _, _ = p.getCameraImage(width=im_w, height=im_h, viewMatrix=view_matrix,
                                                 projectionMatrix=projection_matrix, shadow=0, lightDirection=[1, 1, 1],
                                                 renderer=renderer)
        return rgb_pixels

    def show_image_from_distance(self, obj_id, cam_distance, z_offset=0, y_offset=0, x_offset=0):
        rgbPixels = self.get_image_from_distance(obj_id, cam_distance, z_offset=z_offset, y_offset=y_offset,
                                                 x_offset=x_offset)
        Image.fromarray(rgbPixels[:, :, 0:3]).show()

    def reset(self, new_bead_mass=None):
        self.__init__(visualize=self.visualize, real_init=False, new_bead_mass=new_bead_mass, table=self.table)

    def create_beads(self, color=(0, 0, 1, 1), offset=(0, 0, 0)):
        radius = self.radius  # formerly 0.010
        cup_thickness = k * 0.001

        lower, upper = get_lower_upper(self.cupID)
        buffer = cup_thickness + radius
        lower = np.array(lower) + buffer * np.ones(len(lower))
        upper = np.array(upper) - buffer * np.ones(len(upper))
        limits = zip(lower, upper)
        x_range, y_range = list(limits)[:2]
        z = upper[2] - 0.08
        droplets = [create_sphere(radius, color=color) for _ in range(self.num_droplets)]
        bead_mass = 0.002  # 0.005 actual mass of a kidney bean*2
        for droplet in droplets:
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            set_point(droplet, Point(x, y, z))
            p.changeVisualShape(droplet, -1, rgbaColor=color)
            p.changeDynamics(droplet, -1, mass=bead_mass, lateralFriction=0.2, rollingFriction=0.2,
                             spinningFriction=0.2, restitution=0.7)
        i = 0
        for i, droplet in enumerate(droplets):
            x, y = np.random.normal(0, 1e-3, 2)
            set_point(droplet, Point(x + offset[0], y + offset[1], z + i * (2 * radius + 1e-3)))
        return droplets, z + i * (2 * radius + 1e-3)

    def change_all_bead_dynamics(self, kwargs):
        for droplet in self.droplets:
            p.changeDynamics(droplet, -1, **kwargs)

    def drop_beads_in_cup(self, num_droplets=None):
        if num_droplets is not None:
            self.num_droplets = num_droplets
        offset = p.getBasePositionAndOrientation(self.cupID)[0]
        self.droplets = []
        self.droplet_colors = []
        colors = [(0, 0, 1, 1), (1, 0, 0, 1)]
        for color in colors:
            new_drops, highest_z = self.create_beads(color=color, offset=offset)
            self.droplets += new_drops
            self.droplet_colors += self.num_droplets * [color]
            assert (len(self.droplet_colors) == len(self.droplets))
            time_to_fall = np.sqrt(2 * highest_z / 9.8) + 2.0  # (buffer)
            simulate_for_duration(time_to_fall, dt=1 / 240.0)
        # self.zoom_in_on(self.cupID, k*0.6, z_offset=k*0.1)

    def setup(self, beads=True, cup_offset=(0, 0, 0), new_bead_mass=None, table=False, cup_factor=4):
        if self.real_init:
            # setup world
            self.is_real_time = 0
            p.setRealTimeSimulation(self.is_real_time)
            #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "/home/lagrassa/pour_demo.mp4")
            g = 9.8
            p.setGravity(0, 0, -g)
            if table:
                self.table = p.loadURDF(path + "table/table.urdf", 0, 0, 0, 0, 0, 0.707107, 0.707107)
                self.cupStartPos = (-0.04, -0.10, 0.708)
                self.cupStartPos = (-0.17, 0, 0.6544)
            else:
                self.cupStartPos = (0, 0, 0)
                self.cupStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
            if self.visualize:
                self.cupID = p.loadURDF(path + "urdf/cup/" + self.cup_name, self.cupStartPos, self.cupStartOrientation,
                                        globalScaling=k * cup_factor)
            else:
                self.cupID = p.loadURDF(path + "urdf/cup/" + self.cup_name, self.cupStartPos, self.cupStartOrientation,
                                        globalScaling=k * cup_factor)
                blacken(self.cupID)
            self.cup_constraint = p.createConstraint(self.cupID, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                     [0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1])
            p.changeConstraint(self.cup_constraint, self.cupStartPos, self.cupStartOrientation, maxForce=1000)
            p.changeDynamics(self.cupID, -1, mass=5, lateralFriction=0.1, spinningFriction=0.1, rollingFriction=0.1,
                             restitution=0.7)
            if beads:
                if new_world:
                    self.bullet_id = p.saveState()
                if new_bead_mass is not None:
                    [p.changeDynamics(droplet, -1, mass=float(new_bead_mass), lateralFriction=0.99,
                                      spinningFriction=0.99, rollingFriction=0.99) for droplet in self.droplets]
            # to be realistic
            # p.setTimeStep(1/1200.)
            p.setTimeStep(1 / 300.)
            self.real_init = False
        else:
            self.real_init = True
            p.resetSimulation()
            self.setup(new_bead_mass=new_bead_mass, table=self.table)

    def cup_knocked_over(self, cup=None):
        if cup is None:
            cup = self.cupID
        cupPos, cupQuat = p.getBasePositionAndOrientation(cup)
        roll, pitch, yaw = euler_from_quat(cupQuat)
        thresh = 0.7  # pi/4 plus some
        if abs(roll) > thresh or abs(pitch) > thresh:
            print("CUP KNOCKED OVER")
            return True
        return False

    def zoom_in_on(self, objID, dist=k * 0.7, z_offset=0):
        obj_pos, obj_quat = p.getBasePositionAndOrientation(objID)
        roll, pitch, yaw = euler_from_quat(obj_quat)
        p.resetDebugVisualizerCamera(dist, yaw, roll, obj_pos)

    def top_down_zoom_in_on(self, objID):
        obj_pos, obj_quat = p.getBasePositionAndOrientation(objID)
        roll, pitch, yaw = euler_from_quat(obj_quat)
        p.resetDebugVisualizerCamera(0.5, yaw, -70, obj_pos)

    # p.resetDebugVisualizerCamera(0.5, yaw, roll, objPos)
    def simplify_viz(self):
        features_to_disable = [p.COV_ENABLE_WIREFRAME, p.COV_ENABLE_SHADOWS, p.COV_ENABLE_VR_PICKING,
                               p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                               p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]
        for feature in features_to_disable:
            p.configureDebugVisualizer(feature, 0)


# blackens all links in object
def blacken(obj_id, end_index=None):
    p.changeVisualShape(obj_id, -1, rgbaColor=(0, 1, 0, 0))
    if end_index is None:
        end_index = p.getNumJoints(obj_id)
    for link in range(end_index):
        p.changeVisualShape(obj_id, link, rgbaColor=(0, 1, 0, 0))


def greenen(objID, indices):
    for link in indices:
        p.changeVisualShape(objID, link, rgbaColor=(0, 1, 0, 1))




def parse_tuple(input_tuple):
    list_tuple = input_tuple.split(",")
    # remove first and last parenthesis
    list_tuple[0] = list_tuple[0][1:]
    list_tuple[-1] = list_tuple[-1][:-1]
    return tuple([float(item) for item in list_tuple])


if __name__ == "__main__":
    world = CupWorld(visualize=True, table=True)
    import ipdb; ipdb.set_trace()
    pdb.set_trace()
    world.reset()
    pdb.set_trace()
    world.reset()
