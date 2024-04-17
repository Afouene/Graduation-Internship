from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from setting_the_environment import AUVEnvironment  
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import gym
import os 


models_dir="training/PPO"
logdir="logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)  


env=AUVEnvironment()



checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/26")
eval_callback = EvalCallback(env, best_model_save_path="./logs/best_model",
                             log_path="./logs/results", eval_freq=500)

callback = CallbackList([checkpoint_callback, eval_callback]) 

model = PPO("MlpPolicy",env=env,  n_steps=100 ,verbose=0,gamma=0.99,tensorboard_log=logdir,batch_size=100,ent_coef=0.01,learning_rate=0.0003)
Timesteps=400000
model.learn(total_timesteps=Timesteps,progress_bar=True,callback=callback)
model.save(f"{models_dir}/{Timesteps}")
env.close()

