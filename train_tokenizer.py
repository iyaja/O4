import gym
import compiler_gym  # imports the CompilerGym environments
from compiler_gym.envs.llvm.datasets import CBenchDataset

import numpy as np
import pandas as pd
from torch import nn
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForPreTraining,
    RobertaForSequenceClassification,
)
from transformers import Trainer
from transformers import (
    PreTrainedTokenizerFast,
    BertTokenizerFast,
    RobertaTokenizerFast,
)

import tokenizers
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
from tokenizers.processors import BertProcessing, RobertaProcessing

from datasets import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit, Sequence, Split
from tokenizers.processors import RobertaProcessing


SAMPLES = 64
PHASES = 32
BLACKLIST = [
    "benchmark://cbench-v1/ghostscript",
    "benchmark://cbench-v1/bzip2",
    "benchmark://cbench-v1/jpeg-c",
    "benchmark://cbench-v1/ispell",
]


def sampler(env, samples=SAMPLES, phases=PHASES):
    for benchmark in env.datasets["npb-v0"].benchmarks():
        if benchmark in BLACKLIST:
            continue
        print(benchmark)
        for _ in range(samples):
            env.reset(benchmark=benchmark)
            for phase in range(phases):
                action = env.action_space.sample()
                _, reward, done, info = env.step(action)
                env.action_space.to_string(action)
                if done:
                    break
                yield "\n".join(env.observation["Inst2vecPreprocessedText"])


def main():

    # Create gym environment
    env = gym.make("llvm-ic-v0")

    tokenizer = Tokenizer(WordPiece(vocab=env.inst2vec.vocab))
    tokenizer.pre_tokenizer = CharDelimiterSplit("\n")
    # tokenizer.pre_tokenizer = Split(pattern="[INST]", behavior="removed")
    tokenizer.enable_truncation(512)

    base_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[INST]"]
    inst_tokens = list(env.inst2vec.vocab.keys())
    special_tokens = base_tokens + inst_tokens

    trainer = WordPieceTrainer(
        special_tokens=base_tokens, vocab_size=len(env.inst2vec.vocab)
    )

    tokenizer.train_from_iterator(sampler(env), trainer=trainer)
    tokenizer.post_processor = RobertaProcessing(
        cls=("[CLS]", tokenizer.token_to_id("[CLS]")),
        sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
    )

    tokenizer.save("tokenizer.json")

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json", max_len_single_sentence=512
    )
    fast_tokenizer.add_tokens(env.action_space.names)
    fast_tokenizer.add_special_tokens(
        {
            "cls_token": "[CLS]",
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
        }
    )

    fast_tokenizer.push_to_hub("codebert-llvm-ic-v0")


if __name__ == "__main__":
    main()
