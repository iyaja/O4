import compiler_gym  # imports the CompilerGym environments
import gym
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
)

from datasets import load_dataset

from o4.data import prepare_cost_dataset
from o4.models import CostModelTrainer

SAMPLES = 64
PHASES = 32


def main():

    # Create gym environment
    env = gym.make("llvm-ic-v0")

    # Create tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json", max_len_single_sentence=512
    )
    tokenizer.add_tokens(env.action_space.names)
    tokenizer.add_special_tokens(
        {
            "cls_token": "[CLS]",
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
        }
    )

    # Create model
    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base-mlm", num_labels=1
    )
    model.resize_token_embeddings(len(tokenizer))

    # Preapre datasets
    training_dataset = prepare_cost_dataset(tokenizer, env, data_files='data/npd-v0.csv')
    # eval_dataset = prepare_cost_dataset(tokenizer, env, samples=SAMPLES//8, phases=PHASES)

    # Use the DataCollatorWithPadding for more efficient batched padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="results/cost",
        # learning_rate=2e-5,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=20,
        # weight_decay=0.01,
        report_to="wandb",
        run_name=f"codebert-llvm-ic-{SAMPLES}-{PHASES}",
        # push_to_hub=True,
        hub_model_id="iyaja/codebert-llvm-ic",
    )

    trainer = CostModelTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset["train"],
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    
    model.push_to_hub("codebert-llvm-ic-v0")
    tokenizer.push_to_hub("codebert-llvm-ic-v0")

if __name__ == "__main__":
    main()
