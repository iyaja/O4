from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv


def my_policy(env: LlvmEnv) -> None:
    env.observation_space = "InstCount"  # we're going to use instcount space
    pass  # ... do fun stuff!


if __name__ == "__main__":
    eval_llvm_instcount_policy(my_policy)
