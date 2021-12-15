import logging
from compiler_gym.datasets import benchmark
import pandas as pd
from compiler_gym.envs import LlvmEnv
from datasets import Dataset, load_dataset
from tqdm import tqdm

SAMPLES = 64
PHASES = 32
BLACKLIST = [
    "benchmark://cbench-v1/ghostscript",
    "benchmark://cbench-v1/bzip2",
    "benchmark://cbench-v1/jpeg-c",
    "benchmark://cbench-v1/ispell",
]


def environment_sampler(env: LlvmEnv, samples=SAMPLES, phases=PHASES):
    # benchmark = env.benchmark
    for benchmark in env.datasets["npb-v0"].benchmarks():
        if benchmark in BLACKLIST:
            continue
        benchmark_sampler = tqdm(range(samples))
        benchmark_sampler.set_description(f"Sampling {benchmark}")
        for _ in benchmark_sampler:
            env.reset(benchmark=benchmark)
            for phase in range(phases):
                action = env.action_space.sample()
                _, reward, done, info = env.step(action)
                env.action_space.to_string(action)
                if done:
                    break
                action = env.action_space.to_string(action)
                text = env.observation["Inst2vecPreprocessedText"]
                label = reward
                yield [action] + text, reward


def get_env_samples(env: LlvmEnv, samples=SAMPLES, phases=PHASES):
    env_samples = {"text": [], "label": []}
    for x, y in environment_sampler(env, samples, phases):
        env_samples["text"].append(x)
        env_samples["label"].append(y)
    return env_samples


def make_new_dataset(env: LlvmEnv, samples=SAMPLES, phases=PHASES):
    env_samples = get_env_samples(env, samples, phases)
    df = pd.DataFrame(env_samples).astype("object")
    df.to_pickle("data/cost_dataset.pkl")
    ds = Dataset.from_pandas(df)
    return ds


def prepare_cost_dataset(
    tokenizer, env: LlvmEnv, samples=SAMPLES, phases=PHASES, data_files=None, split='train'
):
    if data_files:
        ds = load_dataset("csv", data_files=data_files)
    else:
        ds = make_new_dataset(env, samples, phases)

    def preprocess(example):
        return tokenizer(
            example["text"],
            is_split_into_words=False,
            padding=True,
            truncation=True,
            max_length=512,
        )

    columns = ["input_ids", "token_type_ids", "attention_mask", "label"]

    tokenized = ds.map(preprocess, batched=True)
    tokenized.set_format(type="torch", columns=columns)

    return tokenized


def prepare_policy_dataset(tokenizer, env: LlvmEnv, samples=SAMPLES, phases=PHASES):

    env_samples = {"text": [], "label": []}

    for _ in range(samples):
        env.reset()
        for _ in range(phases):
            action = env.action_space.sample()
            _, reward, done, info = env.step(action)
            if done:
                break
            text = env.action_space.to_string(action) + env.observation["Ir"]
            label = reward
            env_samples["text"].append(text)
            env_samples["label"].append(label)

    dataset = Dataset.from_dict(env_samples)

    def preprocess(example):
        return tokenizer(example["text"], padding=True, truncation=True)

    tokenized = dataset.map(preprocess, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    return tokenized
