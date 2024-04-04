import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from setting_the_environment import AUVEnvironment
import os

models_dir = "training/PPO"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


env = AUVEnvironment()


def objective(trial):
    learning_rate = trial.suggest_categorical("learning_rate", [1e-06, 5e-06, 1e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01])
    gamma = trial.suggest_float("gamma", 0.8, 0.9997)
    ent_coef = trial.suggest_categorical("ent_coef", [0.001, 0.005, 0.01, 0.05, 0.1])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    model = PPO(
        "MlpPolicy",
        env=env,
        n_steps=100,
        verbose=0,
        gamma=gamma,
        tensorboard_log=logdir,
        batch_size=100,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        clip_range=clip_range
    )
    model.learn(total_timesteps=100000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)

  

    return mean_reward


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best learning rate:", study.best_params["learning_rate"])
print("Best gamma:", study.best_params["gamma"])
print("Best ent_coef:", study.best_params["ent_coef"])
print("Best clip range:", study.best_params["clip_range"])
print("Best mean reward:", study.best_value)