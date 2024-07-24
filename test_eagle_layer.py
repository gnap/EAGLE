from eagle.modeling_eagle import EAGLEModel, EAGLE_Config
from transformers import AutoTokenizer
import os
import torch
import torch.nn.functional as F
from safetensors import safe_open

eagle_path = "/home/gnap/Models/EAGLE-llama2-chat-7B"
base_path = "/home/gnap/Models/Llama-2-7B-Chat-GPTQ"

ipt = "<|im_start|>system\nhello<|im_end|>\n"
# ipt = "hello"
tokenizer = AutoTokenizer.from_pretrained(base_path)
configpath = os.path.join(eagle_path, "config.json")
config = EAGLE_Config.from_pretrained(configpath)

tokens = tokenizer(ipt).input_ids
tokens = torch.unsqueeze(torch.as_tensor(tokens), 0)
print(tokens)
#model_path = "/data1/fanruikai/models/eagele_llama/fp16.bin"
model_path = f"{eagle_path}/pytorch_model.bin"
ea_layer = EAGLEModel(config, bias=True)
ea_layer_state_dict = torch.load(model_path,
                                    map_location="cpu")

ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
hidden_states = torch.zeros(1, tokens.size()[1], 4096)
output = ea_layer.forward(hidden_states, tokens).to(torch.float32)
k = 5 
print(output, output.shape)
#print(output[0][k][0:10])
# print(output[0][0][0:10])

"""
model2 = torch.load(f"{base_path}/pytorch_model-00002-of-00002.bin")
headweight = model2["lm_head.weight"].to(torch.float32)
"""
with safe_open(f"{base_path}/model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       if key == "lm_head.weight":
           headweight = f.get_tensor(key).to(torch.float32)
last_headout = F.linear(output, headweight)

#print(last_headout[0][k][0:10], last_headout.size())
print(last_headout, last_headout.shape)
# print(last_headout[0][18][31990:32000], last_headout.size())
