import torch
import torch.nn.functional as F

# 设置默认设备，之后所有张量自动在此设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"device: {device}")

# 直接从硬盘读取训练好的参数和映射表
print("loading...")
checkpoint = torch.load('model.pt', map_location=device, weights_only=False)

W1 = checkpoint['W1']
b1 = checkpoint['b1']
W2 = checkpoint['W2']
b2 = checkpoint['b2']
s_to_i = checkpoint['s_to_i']
i_to_s = checkpoint['i_to_s']

print(f"The model had been loaded to: {W1.device}")
print("Succeeded\n")

# 采样生成名字
g_gen = torch.Generator(device=device).manual_seed(187)  # Generator也要指定设备
num_names = 10

print(f"The {num_names} names are as follows:\n")
for _ in range(num_names):
    out = []
    ix = s_to_i['.']  # 从起始符 '.' 开始
    
    while True:
        # 逐字预测：tensor自动在GPU上创建
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        
        # 前向传播
        logits1 = xenc @ W1 + b1
        h = torch.tanh(logits1)
        logits2 = h @ W2 + b2
        
        counts = torch.exp(logits2)
        probs = counts / torch.sum(counts, dim=1, keepdims=True)
        
        # 多项式抽样
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g_gen).item()
        
        # 遇到结束符则停止
        if ix == s_to_i['.']:
            break
            
        out.append(i_to_s[ix])
        
    print("".join(out))

print(W1)