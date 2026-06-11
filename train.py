import torch
import torch.nn.functional as F 
import math
torch.set_printoptions(linewidth=300)

# 设置默认设备，之后所有新创建的张量都会在GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)  # PyTorch 2.0+ 推荐方式
print(f"默认设备设置为: {device}")

# 导入与处理数据
file_path = "names.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
data2 = content.splitlines()
data = list(".".join(data2))

# 映射
A_data = ["."] + data
data_B = data + ["."]
set_data = sorted(set(data))
s_to_i = {ch: i for i, ch in enumerate(set_data)}
i_to_s = {i: ch for ch, i in s_to_i.items()}

# 参数设置
g = torch.Generator(device=device).manual_seed(123987)
W1 = torch.randn((27, 27), generator=g) * 0.1
b1 = torch.randn((1, 27), generator=g)
W2 = torch.randn((27, 27), generator=g) * 0.1
b2 = torch.randn((1, 27), generator=g)
print(W1.device)

# 大循环
# 循环内参数
epochs = 100
batchsize = 512  # step参数；设置为设置为 2 的幂次方，适配GPU 的流多处理器调度线程，且符合黄金法则（样本数是特征数的10-100倍）。
Total_len = len(A_data)  # stop参数
loss_list = []  # 增加列表，用于学习率优化
lr = 0.1

for epoch in range(epochs):
    # 小循环
    for start in range(0, Total_len, batchsize):  # 以batchsize为大小，将总长度切割，余数部分转化为小于 batchsize的批次。
        end = start + batchsize
        batchA = A_data[start:end]
        batchB = data_B[start:end]
        # 构建数据集与目标集并完成独热编码
        xs, ys = [], []
        for ch1, ch2 in zip(batchA, batchB):
            ix1 = s_to_i[ch1]
            ix2 = s_to_i[ch2]
            xs.append(ix1)
            ys.append(ix2)    
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        xenc = F.one_hot(xs, num_classes=27).float()
        yenc = F.one_hot(ys, num_classes=27).float()

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

        # 学习率优化器
        losses = 10
        loss_list.append(L.item())  # 将loss值剥离并添加到列表，防止计算图引用
        if len(loss_list)<(losses * 2):
            # 梯度更新
            W2 = W2 - lr * dW2
            W1 = W1 - lr * dW1
            b2 = b2 - lr * db2
            b1 = b1 - lr * db1
        else:
            new = loss_list[losses:]
            old = loss_list[:losses]
            L_new = sum(sorted(new)[1:-1]) / (len(new) - 2)
            L_old = sum(sorted(old)[1:-1]) / (len(old) - 2)
            loss_rate = (L_new - L_old)/(L_old + 1e-8)
            lr = max(lr * (1 - math.tanh(loss_rate) / 2), 1e-5)
            # 梯度更新
            W2 = W2 - lr * dW2
            W1 = W1 - lr * dW1
            b2 = b2 - lr * db2
            b1 = b1 - lr * db1
            
            # 删除最开始的10个old loss
            del loss_list[:losses]
            
    print(f"learning rate: {lr}\n")
    print(f"loss: {L.item():.4f}")
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





