## reorganize ideas from test_gripper.py into more usable form

from __future__ import print_function, division
import numpy as np
import pybullet as p
import pybullet_data
from scipy.interpolate import interp1d
import time
import pdb

from plan_base_motion import plan_base_motion, set_base_values, set_point
from create_box import create_box

## x goes towards bottom right, y goes towards upper right, z is height (upwards from table)

## returns True and makes the move if movement without collision is possible, else returns False
## NOTE: currently doesn't support movement in z-direction
def move(body, constraint, goal_pos, use_gui=False, timestep=0.016667):
        pdb.set_trace()
	base_path = plan_base_motion(body, goal_pos)
	if base_path is not None:
		pos = base_path[0]
		for bq in base_path[1:]:
			_move_helper(body, constraint, pos, bq, use_gui)
			pos = bq
		return True
	return False

def push(body, constraint, start_pos, goal_pos, use_gui=False):
	_move_helper(body, constraint, start_pos, goal_pos, use_gui)

## use_gui set to True means the simulation will run at a speed that makes sense for the GUI;
## otherwise the same thing happens but more immediately
## timestep: seconds between each stepSimulation() call, if using the GUI
## increment: distance through which to move between each call to stepSimulation(), if not using the GUI
def _move_helper(body, constraint, start_pos, goal_pos, use_gui=False, timestep=0.016667, increment=0.002523, verbose=False):
	move_dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))

	if use_gui:
		move_time = (5.0 / 0.8078) * move_dist
		pathgen = interp1d([0., move_time], np.vstack((start_pos, goal_pos)).T, fill_value='extrapolate')

		cnt = 0
		ref_time = time.time()
		last_step_time = ref_time
		while (time.time() < ref_time + move_time + 1):   ## 1 sec pause between each move
			cur_time = time.time() - ref_time
			step_time = time.time() - last_step_time   ## time since last step
			if cur_time < move_time and step_time > timestep:
				pos = pathgen(cur_time)
				_incremental_move(body, constraint, pos)
				last_step_time = time.time()
			if verbose and int(cur_time) > cnt:
				cnt += 1
				print ('obj loc = ', p.getBasePositionAndOrientation(objects[-1]))
	else:
		## split path into desired number of increments
		num_incs = int(move_dist / increment) + 1
		poses = np.vstack(np.linspace(start_pos[i], goal_pos[i], num=num_incs) for i in range(len(start_pos))).T
		print(poses.shape)

		for pos in poses:
			_incremental_move(body, constraint, pos)

def _incremental_move(body, constraint, pos):
	p.changeConstraint(constraint, pos, maxForce=500)
	jointPositions=[ 0., 0.000000, 0., 0.000000 ]
	for jointIndex in range (p.getNumJoints(body)):
		p.resetJointState(body,jointIndex,jointPositions[jointIndex])
		p.setJointMotorControl2(body,jointIndex,p.POSITION_CONTROL,jointPositions[jointIndex],0)
	p.stepSimulation()

if __name__ == '__main__':
	use_gui = True
	logging = False

	if use_gui:
		p.connect(p.GUI)
	else:
		p.connect(p.DIRECT)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.resetSimulation()

	## create ground
	objects = [p.loadURDF("plane.urdf", 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]

	## create gripper
	start_pos = [0.300000,0.300006,0.700000]
	pr2_gripper = p.loadURDF("pr2_gripper.urdf", 0.300000,0.300006,0.700000,-0.000000,-0.000000,-0.000031,1.000000)
	pr2_cid = p.createConstraint(pr2_gripper,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0.300000,0.300006,0.700000])

	## create table
	objects += [p.loadURDF("table/table.urdf", 1.000000,-0.200000,0.000000,0.000000,0.000000,0.707107,0.707107)]

	## create blocks
	objects += [create_box(0.1, 0.1, 0.1, mass=0.01)]
	set_point(objects[-1], (0.9, -0.4, 0.8))

	p.setGravity(0,0,-10)

	p.setRealTimeSimulation(0)
	for i in range(100):
		p.stepSimulation()

	if logging:
		logId = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT,"logs/LOG0001.txt",[objects[-1]])

	goal_pos = [0.650000, -0.4, 0.5]
	push(pr2_gripper, pr2_cid, start_pos, goal_pos, use_gui)
	goal_pos_2 = [0.50000, 0.3, 0.5]
	print(move(pr2_gripper, pr2_cid, goal_pos_2, use_gui))

	if logging:
		p.stopStateLogging(logId)

	# print(p.getBasePositionAndOrientation(pr2_gripper))
	# print(p.getBasePositionAndOrientation(pr2_cid))

	p.disconnect()
