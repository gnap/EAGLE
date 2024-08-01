import torch
torch.set_printoptions(linewidth=160)
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import time

total_tokens = 80
print(f"total_tokens={total_tokens}")
model = EaModel.from_pretrained(
    base_model_path="/home/gnap/Models/Llama-2-7B-Chat-GPTQ",
    ea_model_path="/home/gnap/Models/EAGLE-llama2-chat-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=total_tokens,
    depth=4,
)
model.eval()
# your_message="Hello"
# conv = get_conversation_template("vicuna")
# conv.append_message(conv.roles[0], your_message)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()
# prompt = "[INST] Beijing is the captial [/INST]"
# prompt = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nTell me something about the city Beijing.<|im_end|>\n<|im_start|>assistant\n"
prompt = [
    # dict(role="system", content="You are a helpful assistant."),
    dict(role="user", content="Tell me something about the city Beijing."),
]
prompt = model.tokenizer.apply_chat_template(
    prompt, add_generate_prompt=True, tokenize=False
)
#print(prompt)
input_ids = model.tokenizer([prompt], add_special_tokens=False).input_ids
#print(input_ids)
input_ids = torch.as_tensor(input_ids).cuda()
for i in range(20):
    output_ids = model.eagenerate(input_ids, temperature=0, max_new_tokens=24)
torch.cuda.synchronize()
for x in range(5):
    start = time.time()
    for i in range(10):
        output_ids = model.eagenerate(input_ids, temperature=0, max_new_tokens=24)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"TBT={1000*elapsed/(10 * output_ids.shape[-1]-input_ids.shape[-1])}ms")
#print(output_ids)
output = model.tokenizer.decode(output_ids[0])
print(output)
