import enum


class ApproachType(str, enum.Enum):
    SARSA        = "sarsa"
    DQN          = "dqn"
    ACTOR_CRITIC = "actor_critic"
    PPO          = "ppo"


HAVE_VALUE_FUNCTION = {ApproachType.ACTOR_CRITIC, ApproachType.PPO}
DONT_HAVE_VALUE_FUNCTION = {ApproachType.SARSA, ApproachType.DQN}