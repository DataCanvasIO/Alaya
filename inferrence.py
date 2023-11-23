from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import argparse

parser = argparse.ArgumentParser(description='args')
parser.add_argument('input_file', type=str, help='输入文件')
parser.add_argument('output_file', type=str, help='输出文件')
parser.add_argument('adapter_path', type=str, default=None, help='PEFT文件路径')

args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file
adapter_path = args.adapter_path

model_name = "DataCanvas/Alaya-7B-Base"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if adapter_path:
    model.load_adapter(adapter_path)

eos_token_id = 2
bad_words_ids = 3

gpu_id = '0'

pipe = pipeline('text-generation', 
    model=model, 
    tokenizer=tokenizer, 
    bad_words_ids=[[bad_words_ids]],
    eos_token_id=eos_token_id,
    pad_token_id=eos_token_id,
    device='cuda:'+gpu_id
)

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()
instructions = [line.strip() for line in lines]


def do_inference(instruction):
    PROMPT_FORMAT = '### Instruction:\t\n{instruction}\n\n' 
    prompt = PROMPT_FORMAT.format(instruction=instruction)
    result = pipe(prompt, max_new_tokens=1000, do_sample=True, use_cache=True, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
    flag = '### Output:\t\n'
    try:
        output = result[0]['generated_text'].split(flag)[1].rstrip('\n\n')
    except:
        output = ''
    org_output = result[0]['generated_text']
    return output, org_output

with open(output_file, 'w', encoding='utf-8') as file:
    for ins in instructions:
        response, response_org = do_inference(ins)
        result = {'prompt':ins, 'response':response, 'response_org':response_org}
        print(result)
        json.dump(result, file, ensure_ascii=False)
        file.write('\n')
        
print('All done')
