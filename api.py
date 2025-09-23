from typing import List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError, Field

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Predefined configuration (no environment or TOML loading)
MODEL_NAME = 'Alaya-7B-Chat'
GPU_ID = 0  # used only when CUDA is available
BAD_WORD_TOKEN_ID = 3


# Build model config
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
config.attn_config['attn_impl'] = 'torch'

# Initialize model and tokenizer
# Choose dtype based on CUDA availability (use float32 on CPU for compatibility)
selected_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype=selected_dtype,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="left",
)

# Special token handling
eos_token_id = tokenizer.eos_token_id or getattr(config, 'eos_token_id', 2)
pad_token_id = eos_token_id

# Optional bad words token id (single token) as a predefined constant
bad_words_ids = [[BAD_WORD_TOKEN_ID]]

# Choose device for inference
use_cuda = torch.cuda.is_available()
device_for_pipeline: int = int(GPU_ID) if use_cuda else -1

text_generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    bad_words_ids=bad_words_ids,
    eos_token_id=eos_token_id,
    pad_token_id=pad_token_id,
    device=device_for_pipeline,
)


def _format_prompt(instruction: str, history: Optional[List[str]]) -> str:
    """Format the prompt with optional chat history.

    History is expected to be an even-length list alternating user/assistant messages.
    """
    prompt_user = '### Instruction:\t\n{instruction}\n\n'
    prompt_assistant = '### Output:\t\n{output} </s>'

    history_messages = history or []

    parts: List[str] = []
    for index, message in enumerate(history_messages):
        if index % 2 == 0:
            parts.append(prompt_user.format(instruction=message))
        else:
            parts.append(prompt_assistant.format(output=message))

    parts.append(prompt_user.format(instruction=instruction))
    return ''.join(parts)


def do_inference(
    instruction: str,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    top_p: float = 0.9,
    history: Optional[List[str]] = None,
    max_new_tokens: int = 1024,
) -> Tuple[str, str]:
    prompt = _format_prompt(instruction=instruction, history=history)

    result = text_generator(
        prompt,
        temperature=float(temperature),
        repetition_penalty=float(repetition_penalty),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        do_sample=False if float(temperature) == 0 else True,
        use_cache=True,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    generated_text: str = result[0]['generated_text']
    # Extract only the newly generated portion (after the prompt we fed)
    new_text = generated_text[len(prompt):]

    # Remove a leading "### Output:\t\n" if present
    flag = '### Output:\t\n'
    if new_text.startswith(flag):
        new_text = new_text[len(flag):]

    # Fallback if generation failed to produce usable text
    output_text = new_text if new_text.strip() else '抱歉我不能回答这个问题，请重新输入。'

    return output_text, generated_text


class QuestionAnswerRequest(BaseModel):
    instruction: str
    history: List[str] = Field(default_factory=list)
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 1024


class QuestionAnswerResponse(BaseModel):
    output: str
    org_output: str


@app.route('/health', methods=['GET'])
def health() -> tuple:
    return jsonify({'status': 'ok'}), 200


@app.route('/qa', methods=['POST'])
def question_answer():
    try:
        data = request.get_json(force=True, silent=False) or {}
        req = QuestionAnswerRequest(**data)

        output, org_output = do_inference(
            instruction=req.instruction,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            top_p=req.top_p,
            history=req.history,
            max_new_tokens=req.max_new_tokens,
        )
        resp = QuestionAnswerResponse(output=output, org_output=org_output)
        return jsonify(resp.model_dump()), 200
    except ValidationError as ve:
        return jsonify({'error': 'Invalid request', 'details': ve.errors()}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9975)
