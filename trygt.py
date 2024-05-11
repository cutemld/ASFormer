import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def main():
    # 定義五個label
    labels = ['break', 'rally_start', 'rally_end', 'player_a', 'player_b']

    # 讀取ground truth檔案
    ground_truth_file = '/home/cklin/ASFormer/data/badminton/groundTruth/test_02_14999.txt'
    print(f"Reading ground truth file: {ground_truth_file}")
    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = [line.strip() for line in f]
            print(f"Number of lines in ground truth file: {len(ground_truth_data)}")
    except FileNotFoundError:
        print(f"Error: File '{ground_truth_file}' not found.")
        return

    # 讀取inference結果檔案
    inference_file = '/home/cklin/ASFormer/results/badminton/split_1/test_02_14999'
    print(f"Reading inference result file: {inference_file}")
    try:
        with open(inference_file, 'r') as f:
            lines = f.readlines()
            inference_data = lines[1].strip().split()  # 跳過第一行註解
            print(f"Number of labels in inference result file: {len(inference_data)}")
    except FileNotFoundError:
        print(f"Error: File '{inference_file}' not found.")
        return

    # 確保兩個檔案有相同的資料筆數
    print("Checking if the two files have the same number of data instances...")
    if len(ground_truth_data) == len(inference_data):
        print("The two files have the same number of data instances.")
    else:
        print(f"Error: The two files have different number of data instances ({len(ground_truth_data)} and {len(inference_data)}).")
        return

    total_data = len(ground_truth_data)
    correct_counts = {label: 0 for label in labels}
    ground_truth_labels_all = []
    inference_labels_all = []

    # 比對每一筆資料的label順序
    for i in range(total_data):
        ground_truth_label = ground_truth_data[i]
        inference_label = inference_data[i]

        ground_truth_labels_all.append(ground_truth_label)
        inference_labels_all.append(inference_label)

        # 計算每個label的正確數量
        if ground_truth_label == inference_label:
            correct_counts[ground_truth_label] += 1

    # 計算每個label的準確度
    accuracies = {label: correct_counts[label] / total_data for label in labels}

    # 計算overall accuracy
    overall_correct = sum(correct_counts.values())
    overall_accuracy = overall_correct / (total_data * len(labels))

    # 計算precision, recall, f1-score
    precisions = {label: precision_score(ground_truth_labels_all, inference_labels_all, pos_label=label, average=None)[0] for label in labels}
    recalls = {label: recall_score(ground_truth_labels_all, inference_labels_all, pos_label=label, average=None)[0] for label in labels}
    f1_scores = {label: f1_score(ground_truth_labels_all, inference_labels_all, pos_label=label, average=None)[0] for label in labels}


    # 計算confusion matrix
    conf_matrix = confusion_matrix(ground_truth_labels_all, inference_labels_all, labels=labels)

    # 輸出結果
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    for label in labels:
        print(f"{label} accuracy: {accuracies[label]:.4f}, precision: {precisions[label]:.4f}, recall: {recalls[label]:.4f}, f1-score: {f1_scores[label]:.4f}")

    # 繪製confusion matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

if __name__ == "__main__":
    main()