import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========== 自动收集 data/audio_test 下的所有 wav 文件 ==========
audio_dir = "data/audio_test"
audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

# ========== 模型路径 ==========
base_model_path = "Qwen-Audio-Chat"
lora_model_path = "save/qwen-audio-chat/checkpoint-25800"

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置随机种子
torch.manual_seed(1234)

# ========== 加载模型 ==========
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cuda", trust_remote_code=True).eval()
model = PeftModel.from_pretrained(model, lora_model_path)

# ========== 情绪映射 ==========
affect_map = {
    "ang": "angry",
    "con": "contempt",
    "dis": "disgusted",
    "fea": "fear",
    "hap": "happy",
    "neu": "neutral",
    "sur": "surprise",
    "sad": "sad"
}

# 结果保存目录
save_dir = "results/Test_no_output_emo_25800"
os.makedirs(save_dir, exist_ok=True)

# ========== 处理每个音频文件 ==========
for audio_path in audio_files:
    # 获取音频文件名（不含路径）
    filename = os.path.basename(audio_path).replace(".wav", "")
    
    # 提取情绪代码（假设文件名格式如 M003_ang_3_001.wav）
    parts = filename.split("_")
    emotion_code = parts[1] if len(parts) > 1 else "neu"
    emotion = affect_map.get(emotion_code, "neutral")
    
    # 构造查询
    query = tokenizer.from_list_format([
        {'audio': audio_path},
        {'text': (
            f'For each 16kHz audio, split the waveform into frames of 3200 samples (5 fps). '
            f'Each frame produces a 24-dimensional AU vector, with components AU0 to AU23 representing '
            f'facial muscle activations in this fixed order: AU0 left eye closure; AU1 right eye closure; '
            f'AU2 left lid raise; AU3 right lid raise; AU4 left brow lower; AU5 right brow lower; '
            f'AU6 left brow raise; AU7 right brow raise; AU8 jaw-driven mouth opening; AU9 lower lip slide (left); '
            f'AU10 lower lip slide (right); AU11 left lip corner raise; AU12 right lip corner raise; '
            f'AU13 left lip corner stretch; AU14 right lip corner stretch; AU15 upper lip suck; AU16 lower lip suck; '
            f'AU17 jaw thrust; AU18 upper lip raise; AU19 lower lip depress; AU20 chin raise; AU21 lip pucker; '
            f'AU22 cheek puff; and AU23 nose wrinkle. Each AU value is between 0 and 1 and must be formatted '
            f'to two decimal places (that is, write only the decimal point and two digits—for example, .12 for 0.12). '
            f'For each audio segment, record only the AUs that are activated along with their values; '
            f'for example, [(0, .12), (1, .10)] means only AU0 is 0.12 and AU1 is 0.10 while the others remain untriggered. '
            f'The emotion of the current audio is {emotion}. what is the AU sequence of the current audio?'
        )}
    ])

    # 获取模型输出
    response, _ = model.chat(tokenizer, query=query, history=None)
    
    # 清理模型返回值（去除开头提示信息）
    prefix = "The AU sequence for each frame of audio is:"
    if response.startswith(prefix):
        response = response[len(prefix):]
    
    # 尝试解析 JSON
    try:
        au_sequences = json.loads(response)
    except json.JSONDecodeError:
        au_sequences = response  # 解析失败时直接保存文本结果

    # 格式化 AU 结果
    if isinstance(au_sequences, list):
        formatted_result = "\n".join([json.dumps(frame) for frame in au_sequences])
    else:
        formatted_result = str(au_sequences)

    # 保存到 JSON 文件
    save_path = os.path.join(save_dir, f"{filename}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(formatted_result)

    print(f"Processed: {audio_path} -> {save_path}")



# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig
# from peft import PeftModel
# import torch
# import json
# import os

# # 模型路径
# base_model_path = "Qwen-Audio-Chat"
# lora_model_path = "save/qwen-audio-chat/checkpoint-25900"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 设置随机种子
# torch.manual_seed(1234)

# # 加载 tokenizer 和模型
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cuda", trust_remote_code=True).eval()
# model = PeftModel.from_pretrained(model, lora_model_path)

# # 音频文件列表
# audio_files = [
#     "data/wav_16000/M005_con_3_001.wav",
#     "data/wav_16000/M005_con_3_002.wav",
#     "data/wav_16000/M005_con_3_003.wav",
#     "data/wav_16000/M005_con_3_004.wav",
#     "data/wav_16000/M005_con_3_005.wav",
# ]

# # 情绪映射
# affect_map = {
#     "ang": "angry",
#     "con": "contempt",
#     "dis": "disgusted",
#     "fea": "fear",
#     "hap": "happy",
#     "neu": "neutral",
#     "sur": "surprise",
#     "sad": "sad"
# }

# # 结果保存目录
# save_dir = "results/Test"
# os.makedirs(save_dir, exist_ok=True)

# # 处理每个音频文件
# for audio_path in audio_files:
#     # 获取音频文件名（不含路径）
#     filename = os.path.basename(audio_path).replace(".wav", "")
    
#     # 提取情绪代码（假设格式如 M003_ang_3_001.wav）
#     emotion_code = filename.split("_")[1]
#     emotion = affect_map.get(emotion_code, "neutral")
    
#     # 构造查询
#     query = tokenizer.from_list_format([
#         {'audio': audio_path},
#         {'text': f'For each 16kHz audio, split the waveform into frames of 3200 samples (5 fps). Each frame produces a 24-dimensional AU vector, with components AU0 to AU23 representing facial muscle activations in this fixed order: AU0 left eye closure; AU1 right eye closure; AU2 left lid raise; AU3 right lid raise; AU4 left brow lower; AU5 right brow lower; AU6 left brow raise; AU7 right brow raise; AU8 jaw-driven mouth opening; AU9 lower lip slide (left); AU10 lower lip slide (right); AU11 left lip corner raise; AU12 right lip corner raise; AU13 left lip corner stretch; AU14 right lip corner stretch; AU15 upper lip suck; AU16 lower lip suck; AU17 jaw thrust; AU18 upper lip raise; AU19 lower lip depress; AU20 chin raise; AU21 lip pucker; AU22 cheek puff; and AU23 nose wrinkle. Each AU value is between 0 and 1 and must be formatted to two decimal places (that is, write only the decimal point and two digits—for example, .12 for 0.12). For each audio segment, record only the AUs that are activated along with their values; for example, [(0, .12), (1, .10)] means only AU0 is 0.12 and AU1 is 0.10 while the others remain untriggered. The emotion of the current audio is {emotion}. what is the AU sequence of the current audio?'}
#     ])
    
#     # 获取模型输出
#     response, _ = model.chat(tokenizer, query=query, history=None)
    
#     # 提取 AU 结果（去除开头的 "The AU sequence for each frame of audio is:"）
#     if response.startswith("The AU sequence for each frame of audio is:"):
#         response = response[len("The AU sequence for each frame of audio is:"):] 
    
#     # 格式化 AU 结果，每个 AU 数组占一行
#     au_sequences = json.loads(response)  # 转换为 Python 结构
#     formatted_result = "\n".join([json.dumps(frame) for frame in au_sequences])
    
#     # 保存到 JSON 文件
#     save_path = os.path.join(save_dir, f"{filename}.json")
#     with open(save_path, "w") as f:
#         f.write(formatted_result)
    
#     print(f"Processed: {audio_path} -> {save_path}")
