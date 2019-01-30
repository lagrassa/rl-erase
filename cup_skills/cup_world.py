from __future__ import division
import pybullet as p
import os
import math
import csv
import pdb
from PIL import Image
import numpy as np
import utils
import time
import pybullet_data
k = 1 #scaling factor
DEMO =False 
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat, set_pose, set_joint_positions, get_joint_positions

real_init = True
new_world = True

 
class CupWorld():
    def __init__(self, visualize=False, real_init=True, beads=True, cup_offset=(0,0,0), new_bead_mass = None, table=False, for_pr2 = False):
        proj_dir = os.getcwd()
        if "control" in proj_dir:
            print("Changing working directory to cup skills")
            os.chdir(proj_dir+'/../../../cup_skills')
        
        self.visualize=visualize
        self.real_init = real_init
        self.num_droplets = 100
        self.for_pr2 = for_pr2
        if for_pr2:
            self.radius = k*0.005
        else:
            self.radius = k*0.011
            
        self.table=table
        if real_init:
            try:
                if visualize: #doing this for now to workout this weird bug where the physics doesn't work in the non-GUI version
                    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
                else:
                    physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
            except:
                pdb.set_trace()
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.simplify_viz()
        if for_pr2:
            cup_factor = 1.5
        else:
            cup_factor = 5
        self.setup(beads=True, cup_offset=cup_offset, new_bead_mass = new_bead_mass, table=table,cup_factor = cup_factor)
        if beads:
            pass
            #self.total_bead_mass = self.num_droplets*p.getDynamicsInfo(self.droplets[0], -1)[0]#variable beads would be bad, this is faster 
        

    def toggle_real_time(self):
        self.is_real_time = int(not self.is_real_time)
        p.setRealTimeSimulation(self.is_real_time)

    """saves where all the beads are"""
    def custom_save_beads(self):
        filename = "bead_poses"
        with open(filename, "wb") as csvfile:
            drop_writer = csv.writer(csvfile) 
            for i in range(len(self.droplets)):
                droplet = self.droplets[i]
                color = self.droplet_colors[i]
                pos = p.getBasePositionAndOrientation(droplet)[0]
                drop_writer.writerow([color, pos])

    def custom_restore_beads(self, teleport=False):
        filename = "bead_poses"
        i = 0
        with open(filename, "rb") as csvfile:
            drop_reader = csv.reader(csvfile)
            for row in drop_reader:
                color, pos = [parse_tuple(input_tuple) for input_tuple in row]
                if teleport:
                    set_point(self.droplets[i], pos)
                    i += 1
                else:       
                    create_sphere(self.radius, color=color, pos = pos)
                

    """Returns the proportion of beads that are still in the cup"""
    def ratio_beads_in(self, cup =None):
        if cup is None:
            cup = self.cupID
        aabbMin, aabbMax = p.getAABB(cup)
        all_overlapping =  p.getOverlappingObjects(aabbMin, aabbMax) 
        if all_overlapping is None:
            return 0
        overlapping_objects = [obj for obj in all_overlapping if obj[0] >= self.droplets[0] and obj[0] <= self.droplets[-1]] # in the range to be a droplet
        
        if overlapping_objects is None:
            num_in = 0
        else:
            num_in = len(overlapping_objects)
        total =  self.num_droplets
        return num_in/total

    def distance_from_cup(self, otherObj, otherLinkIndex):
        cupPos = np.array(p.getBasePositionAndOrientation(self.cupID)[0])
        if otherLinkIndex == -1:
            otherPos = np.array(p.getBasePositionAndOrientation(otherObj)[0])
        
        else:
            otherPos = np.array(p.getLinkState(otherObj, otherLinkIndex)[0])
        return np.linalg.norm(cupPos-otherPos)
        
       

    def step(self, timeStep):
        if not self.is_real_time:
            simulate_for_duration(timeStep, dt= 0.01)
        time.sleep(0.0001)
        return 

    """Fun enough, this function returns a tuple of two different camera views!"""
    def world_state(self):
        #crop to only relevant parts
        views = [0, 180] #I have no idea why, but these seems to be in degrees
        images = ()
        for view in views:
            rgbPixels = self.getImageFromDistance(self.cupID, 0.25, z_offset=0.1, theta_offset = view)
            images += (rgbPixels[:,:,0:3],) # decided against cropping 
        return images

    def getImageFromDistance(self, objID,cam_distance, z_offset=0,y_offset=0, x_offset= 0, theta_offset = 0):
        objPos, objQuat = p.getBasePositionAndOrientation(objID)
        adjustedPos = (objPos[0]+x_offset, objPos[1]+y_offset, objPos[2]+z_offset)
        roll, pitch, yaw = euler_from_quat(objQuat)
        yaw = yaw + theta_offset
        cam_distance = k*0.25
        im_w = 50
        im_h = 50
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=adjustedPos, distance=cam_distance, yaw=yaw , pitch=pitch, roll =roll+np.pi, upAxisIndex=2)
        if self.visualize:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER
        fov = 60
        nearPlane = 0.01
        farPlane = 500
        aspect = im_w/im_h
        projectionMatrix = p.computeProjectionMatrixFOV(fov,aspect,nearPlane, farPlane)
  
        _,_,rgbPixels,_,_ = p.getCameraImage(width=im_w,height=im_h, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, shadow=0, lightDirection = [1,1,1],renderer=renderer)
        return rgbPixels
        
        
    def showImageFromDistance(self, objID,cam_distance, z_offset=0, y_offset=0, x_offset = 0):
        rgbPixels = self.getImageFromDistance(objID, cam_distance,z_offset=z_offset, y_offset=y_offset, x_offset=x_offset)
        Image.fromarray(rgbPixels[:,:,0:3]).show()


    def reset(self, new_bead_mass = None):
        self.__init__(visualize=self.visualize, real_init=False, new_bead_mass=new_bead_mass, table=self.table, for_pr2 = self.for_pr2 )
    

    def create_beads(self, color = (0,0,1,1), offset=(0,0,0)):
       radius = self.radius #formerly 0.010
       cup_thickness = k*0.001

       lower, upper = get_lower_upper(self.cupID)
       buffer = cup_thickness + radius
       lower = np.array(lower) + buffer*np.ones(len(lower))
       upper = np.array(upper) - buffer*np.ones(len(upper))
       limits = zip(lower, upper)
       x_range, y_range = limits[:2]
       z = upper[2]-0.08
       droplets = [create_sphere(radius, color=color) for _ in range(self.num_droplets)]
       bead_mass = 0.002 #actual mass of a kidney bean*2
       for droplet in droplets:
           x = np.random.uniform(*x_range)
           y = np.random.uniform(*y_range)
           set_point(droplet, Point(x, y, z))
           p.changeVisualShape(droplet, -1, rgbaColor=color)
           p.changeDynamics(droplet, -1, mass=bead_mass, lateralFriction=0.1, restitution=0.20)

       for i, droplet in enumerate(droplets):
           x, y = np.random.normal(0, 1e-3, 2)
           set_point(droplet, Point(x+offset[0], y+offset[1], z+i*(2*radius+1e-3)))
       return droplets, z+i*(2*radius+1e-3)

    def change_all_bead_dynamics(kwargs):
        for droplet in self.droplets:
            p.changeDynamics(droplet, -1, **kwargs)
            

    def drop_beads_in_cup(self, num_droplets = None):
        if num_droplets is not None:
            self.num_droplets = num_droplets
        offset = p.getBasePositionAndOrientation(self.cupID)[0]
        self.droplets = []
        self.droplet_colors = []
        colors = [(0,0,1,1), (1,0,0,1)]
        for color in colors:
            new_drops, highest_z = self.create_beads(color = color, offset=offset)
            self.droplets += new_drops 
            self.droplet_colors += self.num_droplets*[color]
            assert(len(self.droplet_colors) == len(self.droplets))
            time_to_fall = np.sqrt(2*highest_z/9.8)+3.0 #(buffer)
            simulate_for_duration(time_to_fall, dt= 1/240.0)
        #self.zoom_in_on(self.cupID, k*0.6, z_offset=k*0.1)
        self.custom_save_beads()


    def setup(self, beads=True, cup_offset=(0,0,0), new_bead_mass = None, table=False, cup_factor=5):
        NEW = self.real_init #unfortunately
        if NEW:
            #setup world
            self.is_real_time = 0
            p.setRealTimeSimulation(self.is_real_time)
            g = 9.8
            p.setGravity(0,0,-g)
            if self.visualize:
                self.planeId = p.loadURDF("plane.urdf")
            else:
                self.planeId = p.loadURDF("urdf/invisible_plane.urdf")
                blacken(self.planeId)
            if table:
                self.table = p.loadURDF("table/table.urdf", 0, 0, 0, 0, 0, 0.707107, 0.707107)
                #p.changeDynamics(self.table, -1, lateralFriction=0.99, spinningFriction=0.99, rollingFriction=0.99) 
                self.cupStartPos = (-0.04,-0.10, 0.708)
                self.cupStartPos = (-0.17,0,0.6544)
            else:
                self.cupStartPos = (0,0,0)
                self.cupStartOrientation = p.getQuaternionFromEuler([0,0,0])
            self.cup_name = "cup_pourer.urdf"
            if self.visualize:
                self.cupID = p.loadURDF("urdf/cup/"+self.cup_name,self.cupStartPos, self.cupStartOrientation, globalScaling=k*cup_factor)
            else:
                self.cupID = p.loadURDF("urdf/cup/"+self.cup_name,self.cupStartPos, self.cupStartOrientation, globalScaling=k*cup_factor)
                blacken(self.cupID)
            #self.cup_constraint = p.createConstraint(self.cupID, -1, -1, -1, p.JOINT_FIXED, [0,0,1], [0,0,0], [0,0,0], [0,0,0,1], [0,0,0,1])
            #p.changeConstraint(self.cup_constraint, self.cupStartPos, self.cupStartOrientation)
            p.changeDynamics(self.cupID, -1, mass = 10,  lateralFriction=0.99, spinningFriction=0.99, rollingFriction=0.99, restitution=0.10) 
            if beads:
                if new_world:
                    self.bullet_id = p.saveState()
                else:
                    self.custom_restore_beads()
                if new_bead_mass is not None:
                    [p.changeDynamics(droplet, -1, mass=float(new_bead_mass), lateralFriction=0.99, spinningFriction=0.99, rollingFriction=0.99) for droplet in self.droplets]
                #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "pour_heavy_demo.mp4")
            #to be realistic
            #p.setTimeStep(1/1200.)
            p.setTimeStep(1/200.)
            self.real_init = False
        else:
            self.real_init = True
            p.resetSimulation()
            self.setup(new_bead_mass=new_bead_mass, table = self.table)
    def cup_knocked_over(self, cup=None):
        if cup is None:
            cup = self.cupID
        cupPos, cupQuat =  p.getBasePositionAndOrientation(cup)
        roll, pitch, yaw = euler_from_quat(cupQuat)
        thresh = 0.7 #pi/4 plus some
        if abs(roll) > thresh or abs(pitch) > thresh:
            print("CUP KNOCKED OVER")
            return True
        return False
        


    def zoom_in_on(self,objID, dist = k*0.7, z_offset = 0):
        objPos, objQuat = p.getBasePositionAndOrientation(objID)
        adjustedPos = (objPos[0], objPos[1], objPos[2]+z_offset)
        roll, pitch, yaw = euler_from_quat(objQuat)
        p.resetDebugVisualizerCamera(dist, yaw, roll, objPos)

    def top_down_zoom_in_on(self,objID):
        objPos, objQuat = p.getBasePositionAndOrientation(objID)
        roll, pitch, yaw = euler_from_quat(objQuat)
        p.resetDebugVisualizerCamera(0.5, yaw, -70, objPos)
 
    #p.resetDebugVisualizerCamera(0.5, yaw, roll, objPos)
    def simplify_viz(self):
        features_to_disable = [p.COV_ENABLE_WIREFRAME, p.COV_ENABLE_SHADOWS, p.COV_ENABLE_VR_PICKING, p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]
        for feature in features_to_disable:
            p.configureDebugVisualizer(feature, 0)

#blackens all links in object
def blacken(objID, end_index =None):
    p.changeVisualShape(objID, -1, rgbaColor=(0,1,0,0))
    if end_index is None:
        end_index =  p.getNumJoints(objID)
    for link in range(end_index):
        p.changeVisualShape(objID, link, rgbaColor=(0,1,0,0))
def greenen(objID, indices):
    for link in indices:
        p.changeVisualShape(objID, link, rgbaColor=(0,1,0,1))
    
    

def closest_point_circle(center, xy_pos, radius):
    A = center
    B = xy_pos
    Cx = A[0]+radius*(B[0]-A[0])/np.linalg.norm(B-A)
    Cy = A[1]+radius*(B[1]-A[1])/np.linalg.norm(B-A)
    return np.array([Cx,Cy])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def parse_tuple(input_tuple):
    list_tuple = input_tuple.split(",")
    #remove first and last parenthesis
    list_tuple[0] = list_tuple[0][1:]
    list_tuple[-1] = list_tuple[-1][:-1]
    return tuple([float(item) for item in list_tuple])
        

if __name__ == "__main__":
    world = CupWorld(visualize=True, table=True)
    pdb.set_trace()
    world.reset()
    pdb.set_trace()
    world.reset()
	

