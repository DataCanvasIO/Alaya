from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from datasets import load_dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
import os

dataset = load_dataset("tatsu-lab/alpaca", split="train")

model_name = "DataCanvas/Alaya-7B-Base"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\t\n{example['instruction'][i]}\n\n### Output:\t\n{example['output'][i]} </s>"
        output_texts.append(text)
    return output_texts

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

print_trainable_parameters(model)

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


training_args = TrainingArguments(
    output_dir='./output/',
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy='epoch',
    logging_steps=1,
)
training_args = training_args.set_optimizer(name="adamw_torch", learning_rate=1.0e-5)

trainer = SFTTrainer(
    model,
    args=training_args,
    max_seq_length=2048,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    callbacks=[SavePeftModelCallback()]
)

trainer.train() 

print('Done')