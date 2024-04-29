import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from setting_the_environment import AUVEnvironment


env = AUVEnvironment()


def mask_fn(env: gym.Env) -> np.ndarray:

    available_indices_for_transmission, _ = env.indices_state_for_transmitting()

    # Create the action mask for data transmission
    action_mask = np.zeros(env.num_devices)
    action_mask[available_indices_for_transmission] = 1  

    return env.valid_action_mask()


env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model.learn()

# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
model.predict(observation, action_masks=valid_action_array)