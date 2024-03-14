import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from setting_the_environment import AUVEnvironment  

import os 

models_dir="training/PPO"
logdir="logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)  




env=AUVEnvironment()

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 0.003, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.9997)
    ent_coef = trial.suggest_float("ent_coef", 0, 0.01)
    
    model = PPO("MlpPolicy", env=env, n_steps=100, verbose=0, gamma=gamma, tensorboard_log=logdir, batch_size=100, ent_coef=ent_coef, learning_rate=learning_rate)
    model.learn(total_timesteps=150000)  

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)  
    
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  

print("Best learning rate:", study.best_params["learning_rate"])
print("Best gamma:", study.best_params["gamma"])
print("Best ent_coef:", study.best_params["ent_coef"])
print("Best mean reward:", study.best_value)
