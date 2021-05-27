import pandas as pd


def evaluate(df: pd.DataFrame):
    """
    IPS tutorial:
    https://www.cs.cornell.edu/courses/cs7792/2016fa/lectures/03-counterfactualmodel_6up.pdf
    """

    cum_reward_logging_policy = 0
    cum_reward_new_policy = 0
    for _, row in df.iterrows():
        cum_reward_logging_policy += row["reward"]
        cum_reward_new_policy += (row["new_action_prob"] / row["action_prob"]) * row[
            "reward"
        ]

    return {
        "expected_reward_logging_policy": cum_reward_logging_policy / len(df),
        "expected_reward_new_policy": cum_reward_new_policy / len(df),
    }
