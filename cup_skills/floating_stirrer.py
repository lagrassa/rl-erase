from __future__ import division
from collections import OrderedDict
from gym.spaces import Dict

from cup_skills.cup_world import *
from cup_skills.local_setup import path
from cup_skills.reward import stir_reward
from gym import spaces
from cup_skills.utils import simulate_for_duration, threshold_img

k = 1  # scaling factor
DEMO = False

real_init = True


#was 70
class World:
    def __init__(self, visualize=False, real_init=True, stirring=True, beads=True, num_beads=15, distance_threshold=0.4, states=[[],[]], force_states=[]):
        # make base world
        self.visualize = visualize
        self.distance_threshold = distance_threshold
        self.reward_threshold = 81 #distance_threshold
        self.stirring = stirring
        self.env=self #sorry
        self.unwrapped = self
        self.real_init = real_init
        self.threshold = 0.2  # TAU from thesis
        self.scale = 5
        self.time = 0
        self.side_states, self.top_states = states
        self.force_states = force_states
        if stirring:
            self.state_function = self.stirring_state
        else:
            self.state_function = self.scooping_state
        self.timeout = 20
        self.seed = lambda x: np.random.randint(10)
        self.reward_range = (-100, 100)
        self.reward_scale = 1
        self.metadata = {"threshold": self.threshold}
        if stirring:
            cup_name = "cup_small.urdf"
            bead_radius = 0.011
            camera_z_offset = 0.05
            camera_distance = 0.2
        else:
            cup_name = "cup_3.urdf"
            bead_radius = 0.014
            camera_z_offset = 0.15
            camera_distance = 0.8
        if real_init:
            self.base_world = CupWorld(visualize=visualize, camera_z_offset=camera_z_offset,  bead_radius = bead_radius, real_init=real_init, beads=beads, cup_name = cup_name, camera_distance=camera_distance, for_scoop=not self.stirring)
            self.setup(num_beads=num_beads, scooping_world=not stirring)
         
        self.gen_obs_space()
        self.dt = 0.1
        max_move = 0.8
        low_act = np.array([-max_move] * 7)
        high_act = np.array([max_move] * 7)
        low_act[-1] = -40
        high_act[-1] = 40
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

    def gen_obs_space(self,encoder_dict=None):
        state = self.state_function()
        if isinstance(state, dict):
            ob_space_dict = OrderedDict()
            
            for space_name in state.keys():
                if encoder_dict is not None and "im" in space_name:
                    if "im" in encoder_dict.keys():
                        shape = encoder_dict[space_name].get_output_shape_at(0)[1:]
                    else:
                        shape = state[space_name].shape
                elif encoder_dict is not None and space_name == "forces":
                    if "forces" in encoder_dict.keys():
                        shape = (encoder_dict[space_name].flow[-1].output_dim,)
                    else:
                        shape = state["forces"].shape
                else:
                    shape = state[space_name].shape
                high = np.inf * np.ones(shape)
                low = -high
                ob_space_dict[space_name] = spaces.Box(low, high, dtype=np.float32)
            self.observation_space = Dict(ob_space_dict)
        else:
            high = np.inf * np.ones(state.shape)
            low = -high
            self.observation_space = spaces.Box(low, high, dtype=np.float32)



    # positive when good, negative when bad
    def step(self, action_taken):
        self.time += 1
        self.move_spoon(action_taken[0:-1], max_force=self.scale * action_taken[-1])
        ob = self.state_function()
        if isinstance(ob, tuple):
            reward = ob[-1][-1]
        elif isinstance(ob, dict):
            reward = self.get_scooping_reward()
        elif not self.stirring and len(ob.shape) == 3: # is 3 dimensional
            reward = self.get_scooping_reward()
        else:
            reward = ob[-1]

        # if self.time == self.timeout:
        #    print("action", action)
        #    print("reward", reward_raw)
        done = False
        info = {"is_success": float(reward >= 0)}
        info['r'] = reward
        # info["reward_raw"] = reward_raw
        return ob, reward, done, info

    """try doing what fetchpush does essentially"""

    def move_spoon(self, action, max_force=40):
        pos, orn = p.getBasePositionAndOrientation(self.stirrer_id)

        if len(action) < 6:
            new_orn = orn
        else:
            delta_euler= np.array(action[3:])
            euler = delta_euler + np.array(p.getEulerFromQuaternion(orn))
            new_orn = p.getQuaternionFromEuler(euler)
        new_pos = np.array((pos[:]))
        new_pos += action[0:3]
        if action[2] < 0:
            color = (1,0,0)
        else:
            color = (0,1,0)
        p.addUserDebugLine(pos, new_pos, lineColorRGB=color, lifeTime=1)
        p.changeConstraint(self.cid, new_pos, new_orn, maxForce=max_force)  # 120)
        simulate_for_duration(self.dt)

    def stirring_state(self):
        world_state = self.base_world.world_state()
        stirrer_state = self.stirrer_state()
        # you don't need to worry about features or invariance.....so just make it a row vector and roll with it.
        # return np.hstack([np.array(world_state).flatten(),stirrer_state.flatten()]) #yolo
        reward_raw = stir_reward(world_state, self.base_world.ratio_beads_in_cup())
        reward_for_state = self.reward_scale * (reward_raw - self.reward_threshold)
        return np.hstack([stirrer_state.flatten(), reward_for_state])

    #reward for spoon being out of the cup, nothing otherwise
    def scooping_state(self):
        stirrer_state = self.stirrer_state()
        world_state = self.base_world.world_state()
        reward_for_state,  ratio_beads_in_target, ratio_beads_in_origin= self.get_scooping_reward(ret_tuple=True)
        joint_pos, joint_vel, joint_reactions, _ = p.getJointState(self.stirrer_id, 0)
        if reward_for_state > 0.1:
            print("reward", reward_for_state)
            self.side_states.append(world_state[0])
            self.top_states.append(world_state[1])
            self.force_states.append(np.array(joint_reactions))
            np.save("side_states2.npy",self.side_states)
            np.save("top_states2.npy",self.top_states)
            np.save("force_states2.npy",self.force_states)
        #return world_state, np.hstack([stirrer_state.flatten(), reward_for_state])
        #return world_state[0]
        return OrderedDict({'imside':world_state[0],'imtop':world_state[1], 'forces':np.array(joint_reactions), 'qeulerqdot':stirrer_state})
        #return OrderedDict({'imside':world_state[0],'imtop':world_state[1], 'forces':np.array(joint_reactions)})


    def get_scooping_reward(self, ret_tuple=False):
        #aabbMin, aabbMax = p.getAABB(self.base_world.cupID)
        #all_overlapping = p.getOverlappingObjects(aabbMin, aabbMax)
        #spoon_in_cup = (self.stirrer_id,-1) in all_overlapping
        cup_pos = np.array(p.getBasePositionAndOrientation(self.base_world.cupID)[0])
        scoop_pos = np.array(p.getBasePositionAndOrientation(self.stirrer_id)[0])
        ratio_beads_in_target =  self.base_world.ratio_beads_in_target(self.scoop_target)
        ratio_beads_in_origin =  self.base_world.ratio_beads_in_target(self.base_world.cupID)
        total_beads_accounted_for = ratio_beads_in_target+ratio_beads_in_origin 
        out_penalty = -1*(1-total_beads_accounted_for)
        if ratio_beads_in_target > 0.5:
            print("ratio beads in target", ratio_beads_in_target)
        #world_state = self.base_world.world_state()
        reward_for_state  = 5*ratio_beads_in_target+ out_penalty
        if ret_tuple:
            #print("reward_for_state", reward_for_state) 
            return reward_for_state,ratio_beads_in_target, ratio_beads_in_origin
        else:
            #print("reward_for_state", reward_for_state) 
            return reward_for_state

    def stirrer_far(self):
        dist = self.base_world.distance_from_cup(self.stirrer_id, -1)
        threshold = 0.5
        return dist > threshold

    # this function is now a complete lie and has not only the stirrer state but
    # also the vector from the cup
    # also velocity relative to cup
    def stirrer_state(self):
        # returns position and velocity of stirrer flattened
        # r, theta, z in pos
        cup_pos, cup_quat  = p.getBasePositionAndOrientation(self.base_world.cupID)
        cup_pos = np.array(cup_pos)
        cup_euler = np.array(p.getEulerFromQuaternion(cup_quat))
        stirrer_pos = np.array(p.getBasePositionAndOrientation(self.stirrer_id)[0])

        vector_from_cup = stirrer_pos - cup_pos
        # forces in cup frame
        velocity_vec = np.array(p.getBaseVelocity(self.stirrer_id)[0]) - np.array(
            p.getBaseVelocity(self.base_world.cupID)[0])
        return np.hstack([vector_from_cup, cup_euler, velocity_vec])
    '''
    keep track of period with velocity: go in the direction the velocity is already going but once the pos is
    getting far, reverse it if velocity is low, gain momentum by moving to some random direction does z pid control
    '''
    def manual_stir_policy(self, state):
        state = state["qqdot"]
        pos_vec = state[0:2]
        velocity_vec = state[3:6]
        max_dist = 0.03  # tunable
        slow = 0.01
        max_force = 0.8
        zdes = 0.08
        # print("vel", np.linalg.norm(velocity_vec))
        # print("vel_vec", velocity_vec.round(3))
        if np.linalg.norm(pos_vec) > max_dist:
            velocity_vec = np.hstack([-pos_vec[0:2], 0])
        elif np.linalg.norm(velocity_vec) < slow:
            velocity_vec = np.array([0.1, 0, 0])
        # zpid
        velocity_vec[2] = zdes - state[2]
        overshoot = 10
        dt = overshoot * self.dt
        dpos = dt * velocity_vec
        return np.hstack([dpos, max_force])
    """
    manual scoop policy
    while the reward is low, dip the spoon in the cup at an angle
    bring the spoon out at that same angle once it's in
    """
    def manual_scoop_policy(self, obs_dict):
        imgtop = obs_dict["imtop"]
        imgside = obs_dict["imside"]
        obs = obs_dict["qeulerqdot"]
        forces = obs_dict["forces"]
        spoon_submerged = False
        coming_up = True
        #print(np.round(obs[-1],5))
            
        
        #from PIL import Image
        #Image.fromarray(img).show()
        lower_green = np.array([59,0,0])
        upper_green = np.array([61,400,400])
        spoon_submerged = False
        spoon_img = threshold_img(imgside, lower_green, upper_green)
        beads_img = threshold_img(imgside, np.array([119,0,0]),np.array([122,400,400])) + threshold_img(imgside, np.array([119,0,0]),np.array([150,400,400]))
        green_xs, green_ys = np.nonzero(spoon_img)
        bead_xs, bead_ys = np.nonzero(beads_img)
        euler = obs[3:6]
        euler = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.stirrer_id)[1])
        if len(bead_xs) and len(green_xs) and min(bead_xs) <= max(green_xs):
            spoon_submerged = True


        kp = 0.3
        if spoon_submerged:
            target_z = 0.2
            kp2 = 0.3
            new_pos = [0,0,kp2*(target_z-obs[2])]
        #
            target_euler = [-1.971,0,0]
            target_euler = [-2.7,0,0]
            target_euler = [-2.6,0,0]
            target_euler[1] = euler[1]
            target_euler[2] = euler[2]
            current_z = obs[2]
            current_x = obs[0]
            if abs(current_z-target_z)< 0.03:
                print("at level")
            target_x = 0.4
            new_pos = [kp2*(target_x-current_x),0,kp2*(target_z-current_z)]
            if abs(current_x-target_x) < 0.04:
                target_euler[0] = euler[0]
                print("at loc", euler)
                kp = 0.7
                target_euler[1] = 3*np.pi/4.0

        else:
            new_pos = [0,0,-0.1]
            target_euler = [-2.2,0,0]
            target_euler[1] = euler[1]
            target_euler[2] = euler[2]

        new_euler = kp*np.subtract(target_euler, euler)
        return np.hstack([new_pos, new_euler,2.7])

    def reset(self):
        p.restoreState(self.bullet_id)
        self.__init__(visualize=self.visualize, real_init=False, distance_threshold=self.threshold, stirring = self.stirring, states=(self.side_states, self.top_states), force_states = self.force_states)
        return self.state_function()

    def setup(self, num_beads=2, scooping_world = False):
        start_pos = [0, 0, 0.2]
        start_quat = (0.0, 1, -1, 0.0)
        self.base_world.drop_beads_in_cup(num_beads)
        self.stirrer_id = p.loadURDF(path + "urdf/green_spoon.urdf", globalScaling=1.6, basePosition=start_pos,
                                     baseOrientation=start_quat)
        p.enableJointForceTorqueSensor(self.stirrer_id, 0)
        if scooping_world:
            bowl_start_pos = (self.distance_threshold,0.1,-0.1)
            bowl_start_orn = (0,0,1,0)
            self.scoop_target =  p.loadURDF(path + "urdf/cup/cup_4.urdf", globalScaling=4, basePosition=bowl_start_pos,
                                     baseOrientation=bowl_start_orn)
            p.changeDynamics(self.scoop_target,-1, mass=0)


        self.cid = p.createConstraint(self.stirrer_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0], [0, 0, 0],
                                      [0, 0, 0, 1], [0, 0, 0, 1])
        p.changeConstraint(self.cid, start_pos, start_quat)
        simulate_for_duration(0.005)
        self.base_world.zoom_in_on(self.stirrer_id, 2)
        self.bullet_id = p.saveState()
        self.real_init = False


    def simplify_viz(self):
        features_to_disable = [p.COV_ENABLE_WIREFRAME, p.COV_ENABLE_SHADOWS, p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                               p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]
        for feature in features_to_disable:
            p.configureDebugVisualizer(feature, 0)

    def calibrate_reward(self, control=None):
        # color all droplets randomly
        colors = [(1, 0, 0, 1), (0, 0, 1, 1)]
        if not control:
            for droplet in self.base_world.droplets:
                random_color = colors[np.random.randint(len(colors))]
                p.changeVisualShape(droplet, -1, rgbaColor=random_color)
        reward_raw = stir_reward(self.base_world.world_state(), self.base_world.ratio_beads_in_cup())
        # print("Calibration comlplete. Value was", reward_raw)
        return reward_raw

def run_full_calibration():
    controls = []
    mixed = []
    for i in range(10):
        if i % 2 == 0:
            print("Iter", i)
        world = World(visualize=False, num_beads=num_beads)
        # print("Before mixing", i)
        controls.append(world.calibrate_reward(control=True))
        # print("After mixing")
        mixed.append(world.calibrate_reward(control=False))
        p.disconnect()
    data = {}
    print("controls", controls)
    print("mixed", mixed)
    data["num_beads"] = num_beads
    data["control_mean"] = np.mean(controls)
    data["mixed_mean"] = np.mean(mixed)
    data["control_std"] = np.std(controls)
    data["mixed_std"] = np.std(mixed)

    print("Control")
    print("Standard deviation", np.std(controls))
    print("Mean", np.mean(controls))
    print("Mixed")
    print("Standard deviation", np.std(mixed))
    print("Mean", np.mean(mixed))
    np.save(str(num_beads) + "_reward_calibration_more_samples.npy", data)

def run_policy(policy, world):
    rew_of_rews = []
    for j in range(400):
        rews = 0
        ob = world.state_function()
        for i in range(50):
            action = policy(ob)
            ob, reward, _, _ = world.step(action)
            rews += reward
        rew_of_rews.append(rews)
        world.reset()
        #print("j", j)
        #print(rew_of_rews)

class ScoopWorld(World):
    def __init__(self, **kwargs):
        kwargs['stirring'] = False
        super(ScoopWorld, self).__init__(**kwargs)


if __name__ == "__main__":
    import sys
    num_beads = 150
    if len(sys.argv) > 1:
        num_beads = int(sys.argv[1])
    if "slurm" in sys.argv:
        job_to_num_beads = {1: 50, 2: 80, 3: 110, 4: 140, 5: 170, 6: 200}
        num_beads = job_to_num_beads[int(sys.argv[1])]
    if "calibrate" in sys.argv:
        run_full_calibration()
    else:
        visual = "visual" in sys.argv
        world = World(visualize=visual, num_beads=num_beads, stirring=False, distance_threshold=0.3)
        #world.get_scooping_reward()
        run_policy(world.manual_scoop_policy,world)

