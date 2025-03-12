import os
import json
import numpy as np

def load_json_files(path):
    data = {}
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(path, filename)) as f:
                    content = f.read().strip()
                    lines = content.splitlines()
                    if lines:
                        content = lines[0]
                    content = content.replace('(', '[').replace(')', ']')
                    content = content.replace(',.', ', 0.')  # 修复格式
                    content = content.replace('[,', '[0,')  # 修复格式
                    data[filename] = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {filename}: {e}")
                print(f"Content: {content}")
            except Exception as e:
                print(f"Unexpected error with file {filename}: {e}")
    return data

def calculate_metrics(gt, pred):
    gt_dict = {au[0]: au[1] for au in gt}
    pred_dict = {au[0]: au[1] for au in pred}

    correct_predictions = 0
    total_gt = len(gt_dict)
    total_pred = len(pred_dict)
    mae = 0

    for au in gt_dict:
        if au in pred_dict:
            correct_predictions += 1
            # 计算 MAE
            mae += abs(gt_dict[au] - pred_dict[au])
        else:
            # 没有检测到的 AU 计入绝对误差
            mae += gt_dict[au]  # 若未检测到，视为预测为 0 的情况

    accuracy = correct_predictions / total_gt if total_gt > 0 else 0
    recall = correct_predictions / total_gt if total_gt > 0 else 0
    test_accuracy = correct_predictions / total_pred if total_pred > 0 else 0
    mae = mae / total_gt if total_gt > 0 else 0

    return accuracy, recall, test_accuracy, mae

def main():
    gt_path = 'data/MEAD_AU_Simple_Test_Label'
    pred_path = 'results/Test'
    
    gt_data = load_json_files(gt_path)
    pred_data = load_json_files(pred_path)

    overall_accuracy = []
    overall_recall = []
    overall_test_accuracy = []
    overall_mae = []

    # 创建结果文件
    results_file_path = os.path.join('results', 'MEAD_Test_AU.txt')
    with open(results_file_path, 'w') as results_file:
        for filename in pred_data.keys():
            pred = pred_data[filename]
            gt = gt_data.get(filename, [])

            accuracy, recall, test_accuracy, mae = calculate_metrics(gt, pred)

            overall_accuracy.append(accuracy)
            overall_recall.append(recall)
            overall_test_accuracy.append(test_accuracy)
            overall_mae.append(mae)

            result_line = f'File: {filename}, Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, Test Accuracy: {test_accuracy:.2f}, MAE: {mae:.4f}\n'
            print(result_line.strip())
            results_file.write(result_line)

        overall_avg_accuracy = np.mean(overall_accuracy) if overall_accuracy else 0
        overall_avg_recall = np.mean(overall_recall) if overall_recall else 0
        overall_avg_test_accuracy = np.mean(overall_test_accuracy) if overall_test_accuracy else 0
        overall_avg_mae = np.mean(overall_mae) if overall_mae else 0

        overall_summary = (f'Overall Average Accuracy: {overall_avg_accuracy:.2f}, '
                           f'Overall Average Recall: {overall_avg_recall:.2f}, '
                           f'Overall Average Test Accuracy: {overall_avg_test_accuracy:.2f}, '
                           f'Overall Average MAE: {overall_avg_mae:.4f}\n')

        print(overall_summary.strip())
        results_file.write(overall_summary)

if __name__ == "__main__":
    main()




##############下面的算了 loss，acc，prec 和 recall#############

# import os
# import json
# import numpy as np
# from sklearn.metrics import precision_score, recall_score, accuracy_score

# def load_json_files(path):
#     data = {}
#     for filename in os.listdir(path):
#         if filename.endswith('.json'):
#             try:
#                 with open(os.path.join(path, filename)) as f:
#                     content = f.read().strip()
#                     lines = content.splitlines()
#                     if lines:
#                         content = lines[0]
                    
#                     # 替换为有效的 JSON 格式
#                     content = content.replace('(', '[').replace(')', ']')
                    
#                     # 处理潜在的格式问题
#                     content = content.replace(',.', ', 0.')  # 替换如 0,.33 为 0.33
#                     content = content.replace('[,', '[0,')  # 替换如 [,33] 为 [0,33]
                    
#                     data[filename] = json.loads(content)
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON from file {filename}: {e}")
#                 print(f"Content: {content}")  # 输出内容以帮助调试
#             except Exception as e:
#                 print(f"Unexpected error with file {filename}: {e}")
#     return data

# def calculate_metrics(gt, pred, tolerance=0.05):
#     gt_au = {au[0]: au[1] for au in gt}
#     pred_au = {au[0]: au[1] for au in pred}
    
#     # Check activated AU
#     correct_activations = [1 if au in pred_au else 0 for au in gt_au]
#     predicted_activations = [1 if au in gt_au else 0 for au in pred_au]

#     # Calculate loss
#     loss = 0
#     for au, gt_value in gt_au.items():
#         pred_value = pred_au.get(au, 0)
#         loss += (gt_value - pred_value) ** 2
    
#     # Calculate metrics
#     accuracy = accuracy_score(correct_activations, predicted_activations)
#     precision = precision_score(correct_activations, predicted_activations, zero_division=0)
#     recall = recall_score(correct_activations, predicted_activations, zero_division=0)

#     return loss, accuracy, precision, recall

# def main():
#     gt_path = 'data/MEAD_AU_Simple_Test_Label'
#     pred_path = 'results/Test'
    
#     gt_data = load_json_files(gt_path)
#     pred_data = load_json_files(pred_path)

#     total_loss = 0
#     total_accuracy = []
#     total_precision = []
#     total_recall = []

#     for filename in pred_data.keys():
#         pred = pred_data[filename]
#         gt = gt_data.get(filename, [])
        
#         # 只取 Test 中的数据
#         min_length = min(len(gt), len(pred))
#         gt = gt[:min_length]
#         pred = pred[:min_length]

#         loss, accuracy, precision, recall = calculate_metrics(gt, pred)

#         total_loss += loss
#         total_accuracy.append(accuracy)
#         total_precision.append(precision)
#         total_recall.append(recall)

#         print(f'File: {filename}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

#     if len(pred_data) > 0:  # 确保不为零
#         overall_loss = total_loss / len(pred_data)
#         overall_accuracy = np.mean(total_accuracy)
#         overall_precision = np.mean(total_precision)
#         overall_recall = np.mean(total_recall)

#         print(f'Overall Loss: {overall_loss:.4f}, Overall Accuracy: {overall_accuracy:.4f}, Overall Precision: {overall_precision:.4f}, Overall Recall: {overall_recall:.4f}')
#     else:
#         print("No valid prediction data to calculate overall loss.")

# if __name__ == "__main__":
#     main()