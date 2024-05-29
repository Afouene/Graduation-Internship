from stable_baselines3 import PPO
from stable_baselines3 import A2C

from stable_baselines3.common.evaluation import evaluate_policy
from new_env2d import AUVEnvironment  
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import gym
import os 
import eco2ai

tracker = eco2ai.Tracker(project_name="AUV Path learning for 3d", experiment_description="training the PPO model2d",file_name="2d_10nodes31_.csv")

tracker.start()



models_dir="training/PPO"
logdir="logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)  


#env=AUVEnvironment()

env = make_vec_env(AUVEnvironment, n_envs=8)  # Adjust the number of environments as needed


checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/27")
eval_callback = EvalCallback(env, best_model_save_path="./logs/best_model",
                             log_path="./logs/results", eval_freq=500)

callback = CallbackList([checkpoint_callback, eval_callback]) 

model = PPO("MlpPolicy",env=env,  n_steps=100 ,verbose=0,gamma=0.93,tensorboard_log=logdir,batch_size=100,ent_coef=0.01,learning_rate=0.0003)

Timesteps=15000000
model.learn(total_timesteps=Timesteps,progress_bar=True,callback=callback)
model.save(f"{models_dir}/{Timesteps}")
env.close()

tracker.stop()
