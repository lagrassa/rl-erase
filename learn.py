import random
import pdb
from robot import Robot
#Goal: output a policy (map from state to transition) that 
#maximizes reward
def policy_given_q(q):
    policy = {}
    policy_rewards = {}
    for (s,a) in q:
        if s not in policy:
            policy[s] = a
            policy_rewards[s] = q[(s,a)]
        else:
            if q[(s,a)] > policy_rewards[s]:
		policy[s] = a
		policy_rewards[s] = q[(s,a)]
    return policy

def init_q(states, actions):
    d = {}
    for s in states:
        for a in actions:
            d[(s,a)]= 0
    return d

def best_q(q,s):
    best_a = pick_best_a(q,s)
    return q[(s,best_a)]
    

def update_q(q, robot, a, s_old):
    alpha = 0.1
    discount = 0.9
    reward = robot.reward()
    s_prime = robot.state
    prev_exp =  (1-alpha)*q[(s_old,a)]
    new_exp = alpha*(reward + discount*best_q(q,s_prime))
    q[(s_old, a)] = prev_exp  + new_exp
    return q
    

def pick_action(q,s, actions):
    #with p epsilon pick a random action
    epsilon = 0.1
    if random.random() < epsilon:
        rand_indx = random.randint(0,len(actions)-1)
        random_act = actions[rand_indx]
        #print("picking random action:", random_act)
        return random_act

    else:
        return pick_best_a(q,s)
    #with p 1-epsilon, pick the best a

   
def pick_best_a(q,s): 
    best_val = -100000
    best_a = None
    for (s_prime,a) in q:
        if s_prime != s:
            continue
        if q[(s_prime,a)] > best_val:
            best_val =  q[(s_prime,a)]
            best_a = a
    return best_a
    
    

def q_learn(robot, states, actions):
    q = init_q(states, actions)
    num_steps = 2000
    for i in range(num_steps):
        s_old = robot.state
        a = pick_action(q,robot.state, actions)
        #print("Given q ",q," picking action",a)
        robot.move(a)
        #print("Current state: ",robot.state)
        q = update_q(q,robot,a,s_old)
   
    policy = policy_given_q(q)
    print("The policy is: ",policy)
    return policy

def test_dict(policy):
    numsteps = 10
    robot = Robot()
    acc_rew = 0
    for i in range(numsteps):
        robot.move(policy[robot.state])
        acc_rew += robot.reward()
    return acc_rew

def main():
    states = [0,1,2,3,4,5,6]
    actions = [-1,0,1]
    policy = q_learn(Robot(), states, actions)
    reward = test_dict(policy)
    print "The reward is: " +str(reward) 


main()

    
    

 

