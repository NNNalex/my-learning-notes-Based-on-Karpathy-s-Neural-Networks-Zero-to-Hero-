import torch
import torch.nn.functional as F 
torch.set_printoptions(linewidth=300)

# 设置默认设备，之后所有新创建的张量都会在GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)  # PyTorch 2.0+ 推荐方式

# 性能优化设置
if device.type == 'cuda':
    # 添加非阻塞数据传输
    torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速（Ampere及以上架构）

# 导入与处理数据
file_path = "names.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = list(".".join(f.read().splitlines()))  # 消除中间变量

# 映射
s_to_i = {ch: i for i, ch in enumerate(sorted(set(data)))}  # 消除中间变量
i_to_s = {i: ch for ch, i in s_to_i.items()}

# 参数设置
g = torch.Generator(device=device).manual_seed(123987)
W1 = torch.randn((27, 27), generator=g, dtype=torch.float32)
b1 = torch.randn((1, 27), generator=g, dtype=torch.float32)
W2 = torch.randn((27, 27), generator=g, dtype=torch.float32)
b2 = torch.randn((1, 27), generator=g, dtype=torch.float32)

# 大循环

# 在进入大循环之前（大循环外部）一次性处理完毕
A_data = ["."] + data
data_B = data + ["."]
xs = [s_to_i[ch] for ch in A_data]
ys = [s_to_i[ch] for ch in data_B]
X_all = torch.tensor(xs, dtype=torch.long)
Y_all = torch.tensor(ys, dtype=torch.long)

# 整个数据集的 one_hot 直接在外面做完（如果显存允许）
X_enc_all = F.one_hot(X_all, num_classes=27).float()
Y_enc_all = F.one_hot(Y_all, num_classes=27).float()

# 循环内参数
epochs = 100
batchsize = 512  # step参数；设置为设置为 2 的幂次方，适配GPU的流多处理器调度线程，且符合黄金法则（样本数是特征数的10-100倍）。
Total_len = len(A_data)  # stop参数
lr = 0.1

for epoch in range(epochs):
    # 小循环
    for start in range(0, Total_len, batchsize):
        end = start + batchsize
        # 内部直接切片！没有任何 Python 字符串操作，速度提升百倍
        xenc = X_enc_all[start:end]
        yenc = Y_enc_all[start:end]

        # 前向传播
        logits1 = xenc @ W1 + b1
        h = torch.tanh(logits1)
        logits2 = h @ W2 + b2
        counts = torch.exp(logits2)
        counts_sum = torch.sum(counts, dim=1, keepdims=True)
        counts_sum_inv = (counts_sum + 1e-8) ** -1  # 在counts_sum中加一个极小的常量（如 1e-8）,避免NaN/inf出现
        probs = counts * counts_sum_inv
        # probs = exp_logits2 / torch.sum(exp_logits2, dim=1, keepdims=True)  # 可以写为一行，减少中间变量，提高运行速度，避免NaN/inf出现
        log_probs = -1 * yenc * torch.log(probs)
        sum_loss = torch.sum(log_probs, dim=1)
        L = torch.mean(sum_loss)
        
        # 反向传播
        dL = 1.0
        dsum_loss = dL / xenc.shape[0]
        dlog_probs = dsum_loss * torch.ones_like(log_probs)
        dprobs = dlog_probs * -1 * yenc * (probs) ** -1
        dcounts = dprobs * (counts_sum_inv * torch.ones_like(counts))  # 相当于一次手动广播
        dcounts_sum_inv = torch.sum(dprobs * counts, dim=1, keepdim=True)
        dcounts_sum = dcounts_sum_inv * -1 * ((counts_sum + 1e-8) ** -2)
        dcounts += dcounts_sum * torch.ones_like(counts)  # 相当于一次手动广播
        dlogits2 = dcounts * torch.exp(logits2)
        # dlogits2 = (probs - yenc) / batch_size  # 将 Softmax 和 Cross-Entropy 结合在一起从数学公式上化简，你会发现一个极其优雅的结果
        dW2 = h.T @ dlogits2
        dh = dlogits2 @ W2.T
        db2 = torch.sum(dlogits2, dim=0, keepdim=True)
        dlogits1 = dh * (1 - h**2)
        dW1 = xenc.T @ dlogits1
        db1 = torch.sum(dlogits1, dim=0, keepdim=True)

        # 梯度更新
        W2 = W2 - lr * dW2
        W1 = W1 - lr * dW1
        b2 = b2 - lr * db2
        b1 = b1 - lr * db1

print(f"Finshed! loss: {L.item():.4f}")

# 保存模型参数到硬盘
checkpoint = {
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2,
    's_to_i': s_to_i,   # 顺便把映射表也存进去，防止以后字符顺序变了导致索引对不上
    'i_to_s': i_to_s
}
torch.save(checkpoint, 'model.pt')
print("The results have been successfully exported as the file 'model.pt'!")





