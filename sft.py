from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from datasets import load_dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
import os
from typing import List


# Constants
DATASET_NAME = "tatsu-lab/alpaca"
MODEL_NAME = "DataCanvas/Alaya-7B-Base"
OUTPUT_DIR = './output/'
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 1.0e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
NUM_TRAIN_EPOCHS = 1


def load_data():
    return load_dataset(DATASET_NAME, split="train")


def build_model_and_tokenizer():
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return model, tokenizer


def formatting_prompts_func(example) -> List[str]:
    output_texts: List[str] = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\t\n{example['instruction'][i]}\n\n### Output:\t\n{example['output'][i]} </s>"
        output_texts.append(text)
    return output_texts


def print_trainable_parameters(model) -> None:
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def build_peft_model(model):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)
        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


def build_training_args() -> TrainingArguments:
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_strategy='epoch',
        logging_steps=1,
    )
    training_args = training_args.set_optimizer(name="adamw_torch", learning_rate=LEARNING_RATE)
    return training_args


def main() -> None:
    dataset = load_data()
    model, tokenizer = build_model_and_tokenizer()
    model = build_peft_model(model)
    print_trainable_parameters(model)

    training_args = build_training_args()

    trainer = SFTTrainer(
        model,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        callbacks=[SavePeftModelCallback()],
    )

    trainer.train()
    print('Done')


if __name__ == '__main__':
    main()