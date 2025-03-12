from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import PeftModel
import torch

base_model_path = "Qwen-Audio-Chat"
lora_model_path = "save/qwen-audio-chat/checkpoint-25900"

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cuda", trust_remote_code=True).eval()
model = PeftModel.from_pretrained(model, lora_model_path)

query = tokenizer.from_list_format([
    {'audio': 'data/wav_16000/M003_ang_3_001.wav'}, # Either a local path or an url
    {'text': 'For each 16kHz audio, split the waveform into frames of 3200 samples (5 fps). Each frame produces a 24-dimensional AU vector, with components AU0 to AU23 representing facial muscle activations in this fixed order: AU0 left eye closure; AU1 right eye closure; AU2 left lid raise; AU3 right lid raise; AU4 left brow lower; AU5 right brow lower; AU6 left brow raise; AU7 right brow raise; AU8 jaw-driven mouth opening; AU9 lower lip slide (left); AU10 lower lip slide (right); AU11 left lip corner raise; AU12 right lip corner raise; AU13 left lip corner stretch; AU14 right lip corner stretch; AU15 upper lip suck; AU16 lower lip suck; AU17 jaw thrust; AU18 upper lip raise; AU19 lower lip depress; AU20 chin raise; AU21 lip pucker; AU22 cheek puff; and AU23 nose wrinkle. Each AU value is between 0 and 1 and must be formatted to two decimal places (that is, write only the decimal point and two digits—for example, .12 for 0.12). For each audio segment, record only the AUs that are activated along with their values; for example, [(0, .12), (1, .10)] means only AU0 is 0.12 and AU1 is 0.10 while the others remain untriggered. The emotion of the current audio is angry. what is the AU sequence of the current audio?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print("输入的序列:", query)
print(response)


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# from peft import PeftModel
# import torchaudio

# # 设置路径
# base_model_path = "Qwen-Audio-Chat"
# lora_model_path = "save/qwen-audio-chat/checkpoint-25900"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # **加载基础模型**
# print("加载基础模型...")
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
# )

# # **加载 LoRA 适配器**
# print("加载 LoRA 适配器...")
# model = PeftModel.from_pretrained(model, lora_model_path)

# # **加载分词器和处理器**
# print("加载分词器和处理器...")
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

# print("模型加载完成，准备推理...")

# audio_path = "data/MEAD_Audio_Train/M029_sur_3_017.wav"
# text_input = (
#     "The emotion of the current audio is surprise. What is the AU sequence of the current audio? "
#     "Please output a JSON object with the key 'AU_sequence'."
# )

# # 输出调试信息
# print("Audio input:", audio_path)
# print("Text input:", text_input)

# # 检查音频文件是否存在
# try:
#     waveform, sample_rate = torchaudio.load(audio_path)
#     print("Audio file loaded successfully.")
# except Exception as e:
#     print("Error loading audio file:", e)
#     exit()

# # 使用处理器进行处理
# try:
#     inputs = processor(
#         audio=waveform,
#         text=[text_input],
#         return_tensors="pt"
#     )
#     print("Processor outputs:", inputs)
# except Exception as e:
#     print("Error during processing:", e)
#     raise

# # 将张量移动到对应设备
# inputs = {k: v.to(device) for k, v in inputs.items()}
# model.to(device)

# # 模型生成结果（max_new_tokens 可根据需要调整）
# outputs = model.generate(**inputs, max_new_tokens=100)
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("生成结果：", result)







# import json
# import torch
# import librosa
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# from peft import PeftModel

# #  设置路径
# base_model_path = "Qwen-Audio-Chat"
# lora_model_path = "save/qwen-audio-chat/checkpoint-25900"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # **加载基础模型**
# print("加载基础模型...")
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
# )

# # **加载 LoRA 适配器**
# print("加载 LoRA 适配器...")
# model = PeftModel.from_pretrained(model, lora_model_path)

# # **加载分词器和处理器**
# print("加载分词器和处理器...")
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

# print("模型加载完成，准备推理...")


# def load_audio(audio_path):
#     """加载音频文件并转换为 16kHz 采样率"""
#     audio_waveform, sr = librosa.load(audio_path, sr=16000)
#     return torch.tensor(audio_waveform, dtype=torch.float32), sr


# def transcribe_audio_with_text(audio_path, text_prompt):
#     """输入音频文件路径+文本提示，返回 AU 预测结果"""
    
#     # 1. 加载音频
#     audio_waveform, sr = load_audio(audio_path)

#     # 2. **确保 `text=` 传入 `processor()`**
#     inputs = processor(
#         text=text_prompt,  # ✅ 显式传递 `text=`
#         audio=audio_waveform,  # ✅ 确保 `audio` 正确传入
#         sampling_rate=sr,  # 指定采样率
#         return_tensors="pt"
#     )
#     import ipdb
#     ipdb.set_trace()
#     inputs = {k: v.to(device) for k, v in inputs.items()}  # 移动到 GPU
#     print(inputs.keys())
#     print(inputs['input_ids'].shape)
#     # print(inputs['input_ids'])

#     # 3. 进行推理
#     with torch.no_grad():
#         output_ids = model.generate(**inputs, max_new_tokens=512)  # ✅ 只设置 `max_new_tokens`

#     # 4. 解析输出
#     response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     # 5. **检查 JSON 格式**
#     try:
#         response_json = json.loads(response_text)  
#     except json.JSONDecodeError:
#         response_json = {"error": "Model output is not in expected JSON format.", "raw_output": response_text}

#     return response_json


# # **运行推理**
# audio_path = "data/MEAD_Audio_Train/M029_sur_3_017.wav"
# text_prompt = "The emotion of the current audio is surprise. What is the AU sequence of the current audio?"

# output_au_sequence = transcribe_audio_with_text(audio_path, text_prompt)

# print("\n **输入音频**:", audio_path)
# print(" **模型输出 (AU 序列)**:", json.dumps(output_au_sequence, indent=4))


# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig

# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(1234)

# # 加载基础模型和 LoRA 适配器
# base_model_path = "Qwen-Audio-Chat"
# lora_model_path = "save/qwen-audio-chat/checkpoint-25900"
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     device_map="cuda" if device == "cuda" else "cpu",
#     trust_remote_code=True,
#     bf16=True  # 或者使用 fp16，根据需要
# ).eval()

# # 加载 LoRA 适配器
# from peft import PeftModel
# model = PeftModel.from_pretrained(model, lora_model_path)

# # 如 transformers 版本较低，可能需要设置 generation_config
# model.generation_config = GenerationConfig.from_pretrained(base_model_path, trust_remote_code=True)

# # 构造输入：同时包含音频和文本
# query = tokenizer.from_list_format([
#     {
#         "audio": "data/MEAD_Audio_Train/M029_sur_3_017.wav",
#         "content": "The emotion of the current audio is surprise. What is the AU sequence of the current audio?"
#     }
# ])

# # 进行推理
# response, history = model.chat(tokenizer, query=query, history=None)
# print("模型输出 (AU 序列):", response)
