import os
import json
import glob
import ast

# 音频和标签的路径
AUDIO_DIR = "data/MEAD_Audio_Train"
LABEL_DIR = "data/MEAD_AU_Simple_Train_Label"
OUTPUT_FILE = "data/training_data_simple_au_instrucion2.jsonl"

# 情绪标签映射
EMOTION_MAP = {
    "ang": "angry",
    "con": "contempt",
    "dis": "disgusted",
    "fea": "fear",
    "hap": "happy",
    "neu": "neutral",
    "sur": "surprise",
    "sad": "sad"
}

# 用户消息模板
USER_CONTENT = (
    "For each 16kHz audio, split the waveform into frames of 3200 samples (5 fps). Each frame produces a 24-dimensional AU vector, with components AU0 to AU23 representing facial muscle activations in this fixed order: AU0 left eye closure; AU1 right eye closure; AU2 left lid raise; AU3 right lid raise; AU4 left brow lower; AU5 right brow lower; AU6 left brow raise; AU7 right brow raise; AU8 jaw-driven mouth opening; AU9 lower lip slide (left); AU10 lower lip slide (right); AU11 left lip corner raise; AU12 right lip corner raise; AU13 left lip corner stretch; AU14 right lip corner stretch; AU15 upper lip suck; AU16 lower lip suck; AU17 jaw thrust; AU18 upper lip raise; AU19 lower lip depress; AU20 chin raise; AU21 lip pucker; AU22 cheek puff; and AU23 nose wrinkle. Each AU value is between 0 and 1 and must be formatted to two decimal places (that is, write only the decimal point and two digits—for example, “.12” for 0.12). For each audio segment, record only the AUs that are activated along with their values; for example, [(0, .12), (1, .10)] means only AU0 is 0.12 and AU1 is 0.10 while the others remain untriggered."
    "The emotion of the current audio is {emotion}. "
    "what is the AU sequence of the current audio?"
)

# 读取所有的 .wav 文件
wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))

data_entries = []

for wav_file in wav_files:
    filename = os.path.basename(wav_file)
    base_name, _ = os.path.splitext(filename)
    parts = base_name.split("_")
    if len(parts) < 2:
        continue  # 跳过不符合命名规范的文件

    emotion_abbr = parts[1]  # 获取情绪缩写
    emotion = EMOTION_MAP.get(emotion_abbr, "unknown")

    # 找到对应的 JSON 文件
    json_file = os.path.join(LABEL_DIR, base_name + ".json")
    if not os.path.exists(json_file):
        continue  # 没有对应的 AU 标签文件，跳过

    # # 读取 JSON 数据
    # with open(json_file, "r") as f:
    #     lines = f.readlines()  # 逐行读取
    #     au_data = [json.loads(line.strip()) for line in lines]  # 解析每行 JSON 数组
    #     au_sequence = ",".join([json.dumps(frame) for frame in au_data])  # 逗号分隔

    # 读取 "json" 文件（实际上是 Python 字面量）
    with open(json_file, "r") as f:
        lines = f.readlines()
        # 使用 ast.literal_eval 解析每一行
        au_data_frames = []
        for line in lines:
            line = line.strip()
            # 解析成 Python 对象，比如 [(0,0.09), (1,0.06)]
            frame_data = ast.literal_eval(line)
            # 把元组转列表
            frame_data_list = [list(x) for x in frame_data]
            au_data_frames.append(frame_data_list)

    # 把它转成 JSON 字符串时，可以直接 json.dumps
    au_sequence_json = json.dumps(au_data_frames)

    # # 确保 AU 数据是一个列表
    # if not isinstance(au_data, list):
    #     continue

    # # 格式化 AU 数据，每一帧作为一个 AU 向量
    # au_sequence = ",".join([json.dumps(frame) for frame in au_data])
    
    # 组织数据
    entry = {
        "messages": [
            {"role": "user", "audio": wav_file, "content": USER_CONTENT.format(emotion=emotion)},
            {"role": "assistant", "content": f"The AU sequence for each frame of audio is: {au_sequence_json}"}
        ]
    }
    
    data_entries.append(entry)

# 写入 JSONL 文件
with open(OUTPUT_FILE, "w") as f:
    for entry in data_entries:
        json.dump(entry, f)
        f.write("\n")

print(f"Generated {OUTPUT_FILE} with {len(data_entries)} entries.")
