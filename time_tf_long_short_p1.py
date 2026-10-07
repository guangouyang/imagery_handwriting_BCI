
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import matplotlib.pyplot as plt
import pywt
import math
import gc
import os
from sklearn.model_selection import train_test_split
import random
from google.colab import drive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset = scipy.io.loadmat('1.mat')
raw_data = dataset['data']
label = (dataset['label'] - 1).transpose()

data_full = raw_data[:, 100:300, :]
data_early = raw_data[:, 110:160, :]

indices = np.arange(raw_data.shape[2])
train_idx, test_idx = train_test_split(
    indices, test_size=0.1, stratify=label, random_state=42
)

fs = 100
n_channels, n_times, n_trials = data_full.shape
bands = [(1, 3), (3, 6), (6, 12), (12, 25), (25, 45)]
freqs = 10**np.linspace(0.3, 1.6, 10)
wavelet = 'cmor1.5-1.0'
center_freq = pywt.central_frequency(wavelet)
scales = (center_freq * fs) / freqs

data_spec = np.zeros((n_channels, len(bands), n_times, n_trials), dtype=np.float32)
print("Starting Wavelet Transform for Full Data...")
for i in range(n_channels):
    ch_data = data_full[i, :, :]
    ch_data_t = ch_data.T
    coefs, _ = pywt.cwt(ch_data_t, scales, wavelet, sampling_period=1/fs)
    power = np.abs(coefs)**2
    for j, (fmin, fmax) in enumerate(bands):
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.mean(power[mask, :, :], axis=0)
        data_spec[i, j, :, :] = band_power.T
    del coefs, power
    gc.collect()
data_spec = np.log(data_spec + 1e-10)

X_tf_full = data_spec.transpose(3, 1, 2, 0)
X_ts_full = data_full.transpose(2, 1, 0)[:, np.newaxis, :, :]

X_tf_train, X_tf_test = X_tf_full[train_idx], X_tf_full[test_idx]
X_ts_full_train, X_ts_full_test = X_ts_full[train_idx], X_ts_full[test_idx]
y_train, y_test = label[train_idx], label[test_idx]

t_tf_train = torch.tensor(X_tf_train, dtype=torch.float32).to(device)
t_ts_full_train = torch.tensor(X_ts_full_train, dtype=torch.float32).to(device)
t_y_train = torch.tensor(y_train).to(device)
t_tf_test = torch.tensor(X_tf_test, dtype=torch.float32).to(device)
t_ts_full_test = torch.tensor(X_ts_full_test, dtype=torch.float32).to(device)
t_y_test = torch.tensor(y_test).to(device)

batch_size = 512
loader_train_fusion = DataLoader(TensorDataset(t_tf_train, t_ts_full_train, t_y_train), batch_size=batch_size, shuffle=True)
loader_test_fusion = DataLoader(TensorDataset(t_tf_test, t_ts_full_test, t_y_test), batch_size=batch_size, shuffle=False)

X_ts_early_full = data_early.transpose(2, 1, 0)[:, np.newaxis, :, :]
X_ts_early_train, X_ts_early_test = X_ts_early_full[train_idx], X_ts_early_full[test_idx]

t_ts_early_train = torch.tensor(X_ts_early_train, dtype=torch.float32).to(device)
t_ts_early_test = torch.tensor(X_ts_early_test, dtype=torch.float32).to(device)

loader_train_early = DataLoader(TensorDataset(t_ts_early_train, t_y_train), batch_size=batch_size, shuffle=True)
loader_test_early = DataLoader(TensorDataset(t_ts_early_test, t_y_test), batch_size=batch_size, shuffle=False)

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.causal = True

    def forward(self, x1, x2, x3):
        batch_size, seq_len, _ = x1.shape
        Q = self.wq(x1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x3).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x1.device), diagonal=1)
            scores = scores.masked_fill((1-mask).view(1, 1, seq_len, seq_len).bool() == 0, float('-inf'))
        attention_weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context + x3)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        return x + self.pe(positions)

class TF_conv1(nn.Module):
    def __init__(self):
        super().__init__()
        oc = 64
        k_sizes = (np.array([22, 16, 15, 13, 10, 6])*2)[:5]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=oc, kernel_size=(k, 1), padding='same')
            for k in k_sizes
        ])
        self.channel_projection = nn.Conv2d(in_channels=oc * len(self.convs), out_channels=oc, kernel_size=1)
        self.bn = nn.BatchNorm2d(oc)
        self.relu = nn.ELU()
        self.pool = nn.MaxPool2d((3, 1), ceil_mode=True)
        self.drp = nn.Dropout2d(0.3)
        self.lnorm = nn.LayerNorm([oc, math.ceil(data_full.shape[1]/3), 24])

    def forward(self, x):
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            band_input = x[:, i:i+1, :, :]
            conv_outputs.append(conv(band_input))
        x = torch.cat(conv_outputs, dim=1)
        x = self.channel_projection(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.lnorm(x)
        x = self.drp(x)
        return x

class TF_conv2(nn.Module):
    def __init__(self):
        super().__init__()
        oc, k3 = 64, 7
        self.conv = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=(k3, 24), padding=(3, 0))
        self.bn = nn.BatchNorm2d(oc)
        self.relu = nn.ELU()
        self.pool = nn.AvgPool2d((2, 1), ceil_mode=True)
        self.lnorm = nn.LayerNorm([oc, math.ceil(math.ceil(data_full.shape[1]/3)/2), 1])
        self.drp = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.lnorm(x)
        x = self.drp(x)
        return x

class TF_AssembleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TF_conv1()
        self.pos_embedding = nn.ModuleList([LearnedPositionalEmbedding(64, 100) for _ in range(24)])
        self.att_list = nn.ModuleList([SelfAttention(64, 4) for _ in range(24)])
        self.conv2 = TF_conv2()

    def forward(self, x):
        x = self.conv1(x)
        x1 = torch.zeros_like(x)
        for jj in range(24):
            ebd = self.pos_embedding[jj](x[:, :, :, jj].permute(0, 2, 1))
            x1[:, :, :, jj] = self.att_list[jj](ebd, ebd, ebd).permute(0, 2, 1)
        x = self.conv2(x1)
        return x.squeeze(-1).permute(0, 2, 1)

class TS_full_conv1(nn.Module):
    def __init__(self):
        super().__init__()
        oc, k1, k2 = 64, 15, 1
        self.conv = nn.Conv2d(in_channels=1, out_channels=oc, kernel_size=(k1, k2), padding=(7, 0), stride=(1, 1))
        self.bn = nn.BatchNorm2d(oc)
        self.relu = nn.ELU()
        self.pool = nn.MaxPool2d((3, 1), ceil_mode=True)
        self.drp = nn.Dropout2d(0.3)
        self.lnorm = nn.LayerNorm([oc, math.ceil(data_full.shape[1]/3), 24])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.lnorm(x)
        x = self.drp(x)
        return x

class TS_full_AssembleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TS_full_conv1()
        self.pos_embedding = nn.ModuleList([LearnedPositionalEmbedding(64, 100) for _ in range(24)])
        self.att_list = nn.ModuleList([SelfAttention(64, 4) for _ in range(24)])
        self.conv2 = TF_conv2()

    def forward(self, x):
        x = self.conv1(x)
        x1 = torch.zeros_like(x)
        for jj in range(24):
            ebd = self.pos_embedding[jj](x[:, :, :, jj].permute(0, 2, 1))
            x1[:, :, :, jj] = self.att_list[jj](ebd, ebd, ebd).permute(0, 2, 1)
        x = self.conv2(x1)
        return x.squeeze(-1).permute(0, 2, 1)

class TS_Classifier(nn.Module):
    def __init__(self, ts_net, num_classes=26):
        super().__init__()
        self.ts_net = ts_net
        feature_dim = (math.ceil(math.ceil(data_full.shape[1]/3)/2)) * 64
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.ts_net(x)
        flat = torch.flatten(features, 1)
        return self.fc(flat)

class ImprovedFusedNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.tf_net = TF_AssembleNet()
        self.ts_net = TS_full_AssembleNet()
        self.cross_att_ts_tf = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True, dropout=0.3)
        self.cross_att_tf_ts = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True, dropout=0.3)
        feature_dim = (math.ceil(math.ceil(data_full.shape[1]/3)/2)) * 64
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x_tf, x_ts):
        feat_tf = self.tf_net(x_tf)
        feat_ts = self.ts_net(x_ts)
        tf_attended, _ = self.cross_att_tf_ts(query=feat_tf, key=feat_ts, value=feat_ts)
        ts_attended, _ = self.cross_att_ts_tf(query=feat_ts, key=feat_tf, value=feat_tf)
        feat_tf = feat_tf + tf_attended
        feat_ts = feat_ts + ts_attended
        feat_tf_flat = torch.flatten(feat_tf, 1)
        feat_ts_flat = torch.flatten(feat_ts, 1)
        fused = torch.cat((feat_tf_flat, feat_ts_flat), dim=1)
        return self.classifier(fused)

class EarlyTS_conv1(nn.Module):
    def __init__(self):
        super().__init__()
        oc, k1, k2 = 32, 15, 1
        self.conv = nn.Conv2d(in_channels=1, out_channels=oc, kernel_size=(k1, k2), padding=(7, 0), stride=(1, 1))
        self.bn = nn.BatchNorm2d(oc)
        self.relu = nn.ELU()
        self.pool = nn.AvgPool2d((3, 1))
        self.drp = nn.Dropout2d(0.3)
        self.lnorm = nn.LayerNorm([oc, 50//3, 24])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.lnorm(x)
        x = self.drp(x)
        return x

class EarlyTS_conv2(nn.Module):
    def __init__(self):
        super().__init__()
        oc, k3 = 32, 7
        self.conv = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=(k3, 24), padding=(3, 0))
        self.bn = nn.BatchNorm2d(oc)
        self.relu = nn.ELU()
        self.pool = nn.AvgPool2d((2, 1))
        self.lnorm = nn.LayerNorm([oc, 50//3//2, 1])
        self.drp = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.lnorm(x)
        x = self.drp(x)
        return x

class EarlyTS_AssembleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = EarlyTS_conv1()
        self.pos_embedding = LearnedPositionalEmbedding(32, 50//3)
        self.att_list = nn.ModuleList([SelfAttention(32, 4) for _ in range(24)])
        self.conv2 = EarlyTS_conv2()
        self.linear = nn.Linear((50//3//2)*32, 26)

    def forward(self, x):
        x = self.conv1(x)
        x0 = x
        x1 = torch.zeros_like(x).to(device)
        for jj in range(24):
            ebd = self.pos_embedding(x[:, :, :, jj].permute(0, 2, 1))
            x1[:, :, :, jj] = self.att_list[jj](ebd, ebd, ebd).permute(0, 2, 1)
        x = x1 + x0
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

N_ENSEMBLE = 5
EPOCHS_FUSION = 5
EPOCHS_EARLY_TS = 30

fusion_ensemble_logits = []
early_ts_ensemble_logits = []
true_labels_ensemble = []

print(f"\nStarting Separate Training and Ensemble ({N_ENSEMBLE} iterations)...")

for i in range(N_ENSEMBLE):
    print(f"\n{'='*40}\nIteration {i+1}/{N_ENSEMBLE} \n{'='*40}")

    print(f"\n--- Training Full TS Model ({EPOCHS_EARLY_TS} epochs) ---")
    ts_base_net = TS_full_AssembleNet().to(device)
    ts_classifier = TS_Classifier(ts_base_net).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ts = torch.optim.AdamW(ts_classifier.parameters(), lr=0.0005)

    ts_classifier.train()
    for epoch in range(EPOCHS_EARLY_TS):
        for x_tf, x_ts, y in loader_train_fusion:
            optimizer_ts.zero_grad()
            output = ts_classifier(x_ts)
            loss = criterion(output, y[:, 0])
            loss.backward()
            optimizer_ts.step()

    ts_classifier.eval()
    c1, c2 = 0, 0
    with torch.no_grad():
        for x_tf, x_ts, y in loader_test_fusion:
            out = ts_classifier(x_ts)
            _, predicted = torch.max(out.data, 1)
            c1 += (predicted == y[:, 0]).sum().item()
            c2 += len(y)
    ts_full_acc = c1 / c2
    print(f"Full TS Model Accuracy: {ts_full_acc:.4f}")

    print(f"\n--- Training Fusion Model ({EPOCHS_FUSION} epochs) ---")
    fusion_model = ImprovedFusedNet().to(device)
    fusion_model.ts_net.load_state_dict(ts_base_net.state_dict())
    for param in fusion_model.ts_net.parameters():
        param.requires_grad = False

    optimizer_fusion = torch.optim.AdamW(filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=0.0005, weight_decay=1e-4)
    scheduler_fusion = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fusion, T_max=EPOCHS_FUSION)

    fusion_model.train()
    for epoch in range(EPOCHS_FUSION):
        for x_tf, x_ts, y in loader_train_fusion:
            optimizer_fusion.zero_grad()
            output = fusion_model(x_tf, x_ts)
            loss = criterion(output, y[:, 0])
            loss.backward()
            optimizer_fusion.step()
        scheduler_fusion.step()

    fusion_model.eval()
    member_logits_fusion = []
    member_labels = []
    c1, c2 = 0, 0
    with torch.no_grad():
        for x_tf, x_ts, y in loader_test_fusion:
            out = fusion_model(x_tf, x_ts)
            member_logits_fusion.append(out.cpu())
            member_labels.append(y[:, 0].cpu())
            _, predicted = torch.max(out.data, 1)
            c1 += (predicted == y[:, 0]).sum().item()
            c2 += len(y)
    fusion_acc = c1 / c2
    fusion_ensemble_logits.append(torch.cat(member_logits_fusion, dim=0))
    if i == 0:
      true_labels_ensemble = torch.cat(member_labels, dim=0)
    print(f"Fusion Model Accuracy: {fusion_acc:.4f}")

    print(f"\n--- Training Early TS Model ({EPOCHS_EARLY_TS} epochs) ---")
    early_ts_model = EarlyTS_AssembleNet().to(device)
    optimizer_early_ts = torch.optim.AdamW(early_ts_model.parameters(), lr=0.0005)

    early_ts_model.train()
    for epoch in range(EPOCHS_EARLY_TS):
        for x, y in loader_train_early:
            optimizer_early_ts.zero_grad()
            output = early_ts_model(x)
            loss = criterion(output, y[:, 0])
            loss.backward()
            optimizer_early_ts.step()

    early_ts_model.eval()
    member_logits_early_ts = []
    c1, c2 = 0, 0
    with torch.no_grad():
        for x, y in loader_test_early:
            out = early_ts_model(x)
            member_logits_early_ts.append(out.cpu())
            _, predicted = torch.max(out.data, 1)
            c1 += (predicted == y[:, 0]).sum().item()
            c2 += len(y)
    early_ts_acc = c1 / c2
    early_ts_ensemble_logits.append(torch.cat(member_logits_early_ts, dim=0))
    print(f"Early TS Model Accuracy: {early_ts_acc:.4f}")

    member_combined_logits = fusion_ensemble_logits[-1] + early_ts_ensemble_logits[-1]
    _, member_combined_predicted = torch.max(member_combined_logits, 1)
    member_combined_acc = (member_combined_predicted == true_labels_ensemble).sum().item() / len(true_labels_ensemble)
    print(f"Combined (Fusion + Early TS) Accuracy for Iteration {i+1}: {member_combined_acc:.4f}")

print(f"\n{'='*40}\nFinal Results (Logits Summation)\n{'='*40}")

avg_fusion_logits = torch.mean(torch.stack(fusion_ensemble_logits), dim=0)
avg_early_ts_logits = torch.mean(torch.stack(early_ts_ensemble_logits), dim=0)

combined_logits = avg_fusion_logits + avg_early_ts_logits

_, final_predicted = torch.max(combined_logits, 1)
final_acc = (final_predicted == true_labels_ensemble).sum().item() / len(true_labels_ensemble)

print(f"Final Overall Combined Ensemble Accuracy: {final_acc:.4f}")