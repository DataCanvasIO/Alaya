from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import argparse
from typing import Tuple


MODEL_NAME = "DataCanvas/Alaya-7B-Chat"
EOS_TOKEN_ID = 2
BAD_WORD_TOKEN_ID = 3
DEFAULT_GPU_ID = '0'
PROMPT_TEMPLATE = '### Instruction:\t\n{instruction}\n\n'
OUTPUT_FLAG = '### Output:\t\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('input_file', type=str, help='输入文件')
    parser.add_argument('output_file', type=str, help='输出文件')
    parser.add_argument('adapter_path', type=str, default=None, help='PEFT文件路径')
    return parser.parse_args()


def build_pipeline(adapter_path: str, gpu_id: str = DEFAULT_GPU_ID):
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'torch'
    # config.max_seq_len = 4096  # (input + output) tokens can now be up to 4096

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if adapter_path:
        model.load_adapter(adapter_path)

    text_gen_pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        bad_words_ids=[[BAD_WORD_TOKEN_ID]],
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=EOS_TOKEN_ID,
        device='cuda:' + gpu_id,
    )
    return text_gen_pipe


def do_inference(instruction: str, text_gen_pipe, eos_token_id: int) -> Tuple[str, str]:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    result = text_gen_pipe(
        prompt,
        max_new_tokens=1000,
        do_sample=True,
        use_cache=True,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    try:
        output = result[0]['generated_text'].split(OUTPUT_FLAG)[1].rstrip('\n\n')
    except Exception:
        output = ''
    org_output = result[0]['generated_text']
    return output, org_output


def main() -> None:
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file
    adapter_path = args.adapter_path

    gpu_id = DEFAULT_GPU_ID
    text_gen_pipe = build_pipeline(adapter_path=adapter_path, gpu_id=gpu_id)

    with open(input_file, 'r', encoding='utf-8') as file:
        instructions = [line.strip() for line in file.readlines()]

    with open(output_file, 'w', encoding='utf-8') as file:
        for ins in instructions:
            response, response_org = do_inference(ins, text_gen_pipe, EOS_TOKEN_ID)
            result = {'prompt': ins, 'response': response, 'response_org': response_org}
            print(result)
            json.dump(result, file, ensure_ascii=False)
            file.write('\n')

    print('All done')


if __name__ == '__main__':
    main()
