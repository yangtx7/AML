import matplotlib.pyplot as plt

# 假设数据
sequence_lengths = [10, 20, 30, 40, 50]  # 序列长度
normal_attention_times = [0.1, 0.2, 0.3, 0.5, 0.7]  # 正常 Attention 的推理时间
flash_attention_times = [0.05, 0.1, 0.15, 0.3, 0.45]  # Flash Attention 2 的推理时间


len1 = [27,20,176,182,192,314,273,266,291,347,209]
tim1 = [0.8032,0.5927,5.2858,5.4333,5.7381,9.4479,8.1552,7.9407,8.7185,10.3701,6.2336]
paired = list(zip(len1, tim1))
sorted_paired = sorted(paired, key=lambda x: x[0])
sorted_len1, sorted_tim1 = zip(*sorted_paired)
sorted_len1 = list(sorted_len1)
sorted_tim1 = list(sorted_tim1)


len2 = [24,33,224,151,195,283,366,272,495,285,211]
tim2 = [0.7956,1.0901,7.212,4.5555,5.8929,8.6052,11.1662,8.2645,15.1279,8.6768,6.4043]
paired = list(zip(len2, tim2))
sorted_paired = sorted(paired, key=lambda x: x[0])
sorted_len2, sorted_tim2 = zip(*sorted_paired)
sorted_len2 = list(sorted_len2)
sorted_tim2 = list(sorted_tim2)

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制折线图
plt.plot(sorted_len1, sorted_tim1, label='Vanilla Attention', marker='o', linestyle='-', color='blue')
plt.plot(sorted_len2, sorted_tim2, label='Flash Attention 2', marker='o', linestyle='-', color='red')
plt.title('Inference Time Comparison: Vanilla Attention vs Flash Attention 2')
plt.xlabel('Sequence Length')
plt.ylabel('Inference Time (s)')
plt.legend()
plt.grid(True)
plt.savefig("a.pdf")
