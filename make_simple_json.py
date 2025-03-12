import os
import json

def format_value(val):
    """
    将浮点数 round 到 2 位小数后:
      1) 如果四舍五入结果是 0，则返回 None 表示跳过；
      2) 如果结果 >= 1，则保留整数部分并保留到小数点后 2 位（如果需要）；
      3) 如果结果介于 0 和 1 之间，则去掉整数部分的 0（例 0.93 -> .93）。
    """
    val_rounded = round(val, 2)
    if val_rounded == 0:
        return None  # 跳过 0
    
    # 使用字符串格式化保留两位小数
    val_str = f"{val_rounded:.2f}"  # 例如 "0.10", "1.23", "0.93"
    
    # 如果形如 "0.xx" 则去掉 '0'
    if val_str.startswith("0."):
        val_str = "." + val_str[2:]
    elif val_str.startswith("-0."):
        # 如果可能出现负数，可以用这种方式处理
        val_str = "-." + val_str[3:]
    
    return val_str

def process_line(line):
    """
    解析一行形如 [0.1291, 0.0909, ..., 0.0259] 的字符串，
    返回处理后形如 [(3,.32),(7,.81), ...] 的字符串（不含引号）。
    """
    line = line.strip()
    if not line:
        return None
    
    # 利用 json.loads 解析这一行的方括号数组
    # 得到一个长度为 24 的浮点数列表
    row_data = json.loads(line)  # 例如 [0.1291, 0.0909, 0.0, ...]
    
    # 对该行的 24 个浮点数逐个做处理
    processed_tuples = []
    for idx, val in enumerate(row_data):
        formatted_val = format_value(val)
        if formatted_val is not None:
            processed_tuples.append((idx, formatted_val))
    
    # 把元组列表转成目标格式的字符串，例如：
    # [(3,.32),(7,.81),(11,.10),(14,.39)]
    # 注意这里要自己拼字符串，因为要去掉引号，用小括号包住
    # 形如  (3,.32) 这样的元素之间用逗号相连
    tuple_strs = []
    for (i, v) in processed_tuples:
        # 直接拼成 (索引,数值)
        # 数值如 .93 不加引号，保留小数点后两位
        tuple_strs.append(f"({i},{v})")
    
    # 拼成一个大列表：[(0,.12),(3,.56),...]
    line_str = "[" + ",".join(tuple_strs) + "]"
    return line_str

def process_au_file(input_file_path, output_file_path):
    """
    对文件做如下处理：
    1) 每隔 5 行（0,5,10,15, ...）读取一行；
    2) 对这一行做浮点数清洗、去 0、转成 (索引, 值)；
    3) 写到 output 文件里，每行一个结果。
    """
    # 先读入所有行
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 打开输出文件
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        # 每隔 5 行取一行
        for i in range(0, len(lines), 5):
            line_str = process_line(lines[i])
            if line_str is not None:
                out_f.write(line_str + "\n")  # 写完后换行

def main():
    input_dir = 'data/MEAD_AU_Test_label'
    output_dir = 'data/MEAD_AU_Simple_Test_Label'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_au_file(input_file, output_file)
            print(f"Processed {filename}.")

if __name__ == "__main__":
    main()
