from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from setting_the_environment import AUVEnvironment  
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

import os 

models_dir="training/PPO"
logdir="logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)  


env=AUVEnvironment()
#stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=1000, min_evals=5, verbose=0)
#eval_callback = EvalCallback(env, eval_freq=100, callback_after_eval=stop_train_callback, verbose=100)
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/models")
eval_callback = EvalCallback(env, best_model_save_path="./logs/best_model",
                             log_path="./logs/results", eval_freq=500)

callback = CallbackList([checkpoint_callback, eval_callback])

lr=0.0008071712370237489
gamma=0.85175431065817
ent_coef=0.0014511354831551549
model = PPO("MlpPolicy",env=env,  n_steps=100 ,verbose=0,gamma=0.99,tensorboard_log=logdir,batch_size=100,ent_coef=0.01,learning_rate=0.0003)
#model = PPO("MlpPolicy",env=env,  n_steps=100 ,verbose=0,gamma=gamma,tensorboard_log=logdir,batch_size=100,ent_coef=ent_coef,learning_rate=lr)

Timesteps=200000
model.learn(total_timesteps=Timesteps,progress_bar=True,callback=callback)
model.save(f"{models_dir}/{Timesteps}")
env.close()
