import torch
from  transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify
from pydantic import BaseModel

app = Flask(__name__)

model_name = 'Alaya-7B-Chat'
eos_token_id = 2
bad_words_ids = 3

gpu_id = '0'

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'torch'

model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# config.attn_config['attn_impl'] = 'triton'
# config.init_device = 'cuda:' + gpu_id    # For fast initialization directly on GPU!

# config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096


model = AutoModelForCausalLM.from_pretrained(
  model_name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True, 
)

tokenizer = AutoTokenizer.from_pretrained(model_name, local_file_only=True, padding_side="left")

pipe = pipeline('text-generation', 
    model=model, 
    tokenizer=tokenizer, 
    bad_words_ids=[[bad_words_ids]],
    eos_token_id=eos_token_id,
    pad_token_id=eos_token_id,
    device='cuda:'+gpu_id
)


# history=[] 元素数量必须是偶数
def do_inference(instruction, temperature=float(1.0), repetition_penalty=float(1.0), top_p=float(0.9), history=[]):
    PROMPT_FORMAT = '### Instruction:\t\n{instruction}\n\n' 
    OUTPUT_FORMAT = '### Output:\t\n{output} </s>'

    prompt = PROMPT_FORMAT.format(instruction=instruction)
    
    history2llm = []
    
    for i,msg in enumerate(history):
        if i%2==0:  # user
            msg2llm =  PROMPT_FORMAT.format(instruction=msg)
        else: # alaya
            msg2llm =  OUTPUT_FORMAT.format(output=msg)
        history2llm.append(msg2llm)
        
    history2llm_str = ''.join(history2llm)

    prompt2LLM = history2llm_str + prompt 
        
    result = pipe(
        prompt2LLM, 
        temperature=float(temperature),
        repetition_penalty=float(repetition_penalty),
        top_p=float(top_p),
        max_length=2048,
        max_new_tokens=1024, 
        do_sample=False if float(temperature)==0 else True, 
        use_cache=True, 
        eos_token_id=eos_token_id, 
        pad_token_id=eos_token_id
    )
    
    flag = '### Output:\t\n'

    try:
        output = result[0]['generated_text'][len(prompt2LLM):].lstrip(flag)
    except:
        output = '抱歉我不能回答这个问题，请重新输入。'
        
    org_output = result[0]['generated_text']
    
    return output, org_output


class QuestionAnswerRequest(BaseModel):
    instruction: str
    history: list
    temperature: float
    repetition_penalty: float
    top_p: float
    
    
class QuestionAnswerResponse(BaseModel):
    output: str
    org_output: str
    
@app.route('/qa', methods=['POST'])
def question_answer():
    try:
        data = request.get_json()
        request_data = QuestionAnswerRequest(**data)
        instruction = request_data.instruction
        history = request_data.history
        temperature = request_data.temperature
        top_p = request_data.top_p
        repetition_penalty = request_data.repetition_penalty
        output, org_output = do_inference(instruction, temperature, repetition_penalty, top_p, history)
        response_data = QuestionAnswerResponse(output=output, org_output=org_output)
        return jsonify(response_data.model_dump())
    except Exception as e:
        return jsonify({'error': str(e)})
    
    
if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=9975)
