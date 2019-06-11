from __future__ import division

from cup_skills.cup_world import *
from cup_skills.local_setup import path
from cup_skills.reward import stir_reward
from gym import spaces
from cup_skills.utils import simulate_for_duration

k = 1  # scaling factor
DEMO = False

real_init = True


class World:
    def __init__(self, visualize=False, real_init=True, stirring=True, beads=True, num_beads=70, distance_threshold=81):
        # make base world
        self.visualize = visualize
        self.unwrapped = self
        self.real_init = real_init
        self.threshold = distance_threshold  # TAU from thesis
        self.time = 0
        if stirring:
            self.state_function = self.stirring_state
        else:
            self.state_function = self.scooping_state
        self.timeout = 20
        self.seed = lambda x: np.random.randint(10)
        self.reward_range = (-100, 100)
        self.reward_scale = 1
        self.metadata = {"threshold": self.threshold}
        if real_init:
            self.base_world = CupWorld(visualize=visualize, real_init=real_init, beads=beads)
            self.setup(num_beads=num_beads)
        high = np.inf * np.ones(self.state_function().shape[0])
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.dt = 0.1
        max_move = 0.8
        low_act = np.array([-max_move] * 4)
        high_act = np.array([max_move] * 4)
        self.scale = 5  # 45.
        low_act[3] = 8. / self.scale
        low_act[3] = -40. / self.scale  # for ddpg
        high_act[3] = 40 / self.scale
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

    # positive when good, negative when bad
    def step(self, action_taken):
        self.time += 1
        self.move_spoon(action_taken[0:3], max_force=self.scale * action_taken[3])
        ob = self.state_function()
        reward = ob[-1]
        # if self.time == self.timeout:
        #    print("action", action)
        #    print("reward", reward_raw)
        done = not reward < self.threshold or self.time > self.timeout or \
               self.base_world.cup_knocked_over() or self.stirrer_far()
        info = {"is_success": float(reward >= 0)}
        # info["reward_raw"] = reward_raw
        return ob, reward, done, info

    """try doing what fetchpush does essentially"""

    def move_spoon(self, action, max_force=40):
        pos, orn = p.getBasePositionAndOrientation(self.stirrer_id)
        new_pos = np.array((pos[:]))
        new_pos += action
        p.changeConstraint(self.cid, new_pos, orn, maxForce=max_force)  # 120)
        simulate_for_duration(self.dt)

    def stirring_state(self):
        world_state = self.base_world.world_state()
        stirrer_state = self.stirrer_state()
        # you don't need to worry about features or invariance.....so just make it a row vector and roll with it.
        # return np.hstack([np.array(world_state).flatten(),stirrer_state.flatten()]) #yolo
        reward_raw = stir_reward(world_state, self.base_world.ratio_beads_in_cup())
        reward_for_state = self.reward_scale * (reward_raw - self.threshold)
        return np.hstack([stirrer_state.flatten(), reward_for_state])

    #reward for spoon being out of the cup, nothing otherwise
    def scooping_state(self):
        aabbMin, aabbMax = p.getAABB(self.base_world.cupID)
        all_overlapping = p.getOverlappingObjects(aabbMin, aabbMax)
        spoon_in_cup = (self.stirrer_id,-1) in all_overlapping
        if spoon_in_cup:
            reward_for_state = -1
        else:
            print("out of scoop")
            ratio_beads_in_scoop =  self.base_world.ratio_beads_in_scoop(self.stirrer_id)
            #world_state = self.base_world.world_state()
            reward_for_state = self.reward_scale * (ratio_beads_in_scoop - self.threshold)
        stirrer_state = self.stirrer_state()
        return np.hstack([stirrer_state.flatten(), reward_for_state])

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
        cup_pos = np.array(p.getBasePositionAndOrientation(self.base_world.cupID)[0])
        stirrer_pos = np.array(p.getBasePositionAndOrientation(self.stirrer_id)[0])
        vector_from_cup = stirrer_pos - cup_pos
        # forces in cup frame
        velocity_vec = np.array(p.getBaseVelocity(self.stirrer_id)[0]) - np.array(
            p.getBaseVelocity(self.base_world.cupID)[0])
        return np.hstack([vector_from_cup, velocity_vec])

    # keep track of period with velocity: go in the direction the velocity is already going but once the pos is
    # getting far, reverse it if velocity is low, gain momentum by moving to some random direction does z pid control
    def manual_policy(self, state):
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

    def reset(self):
        p.restoreState(self.bullet_id)
        self.__init__(visualize=self.visualize, real_init=False, distance_threshold=self.threshold)
        return self.state()

    def setup(self, num_beads=2):
        start_pos = [0, 0, 0.2]
        start_quat = (0.0, 1, -1, 0.0)
        self.base_world.drop_beads_in_cup(num_beads)
        self.stirrer_id = p.loadURDF(path + "urdf/green_spoon.urdf", globalScaling=1.6, basePosition=start_pos,
                                     baseOrientation=start_quat)
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
    for i in range(30):
        if i % 10 == 0:
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

def run_manual_stir():
    width = 0.4
    force = 0.7
    rew_of_rews = []
    for j in range(40):
        rews = []
        ob = world.state()
        for i in range(60):
            action = world.manual_policy(ob)
            ob, reward, _, _ = world.step(action)
            rews.append(reward)
        rew_of_rews.append(rews)
        print("j", j)
    np.save("rew_of_rews.npy", rew_of_rews)

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
        world = World(visualize=visual, num_beads=num_beads, stirring=False, distance_threshold=1)
        actions = ([0,0,0.2,0.5],[0,0,0.2,0.5],[0,0,0.2,0.5])
        for action in range(0,10):
            action = [0,0,0.1,0.8]
            print(world.step(action))
