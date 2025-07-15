import numpy as np
import csv
from utils import load_test_data, normalize_with, load_model

# 1. 加载测试数据与模型参数
test_x = load_test_data()
w, mean_x, std_x = load_model()

# 2. 标准化 + 添加偏置项
test_x = normalize_with(test_x, mean_x, std_x)
test_x = np.concatenate((np.ones([test_x.shape[0], 1]), test_x), axis=1)

# 3. 预测
ans_y = np.dot(test_x, w)
print("预测完成，结果示例：")
print(ans_y[:5])

# 4. 输出 CSV
with open("submit.csv", mode="w", newline="") as submit_file:
    writer = csv.writer(submit_file)
    writer.writerow(["id", "value"])
    for i in range(240):
        writer.writerow([f"id_{i}", ans_y[i][0]])
print("预测结果已保存为 submit.csv")
