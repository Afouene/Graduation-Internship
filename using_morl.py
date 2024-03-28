
from setting_the_environment import AUVEnvironment  

import numpy as np
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL



env = AUVEnvironment()

agent = PQL(
    env=env,
    ref_point=np.array([0, -50]),  # used to compute hypervolume
    log=True,  # use weights and biases to see the results!
)

agent.train(total_timesteps=100000, eval_env=env, ref_point=np.array([0, -50]))