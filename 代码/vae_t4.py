# @Author : LiZhongzheng
# 开发时间  ：2024-12-31 16:52
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


# 数据加载函数
def load_data(path, filename):
    with open(os.path.join(path, filename), 'rb') as file:
        data = pickle.load(file)
        data = data['data']
        data = np.array(data['observations'])
        return data


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 假设输入数据在0到1之间
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)

        # 解码
        decoded = self.decoder(z)
        return decoded, mu, logvar


# 损失函数
def loss_function(recon_x, x, mu, logvar, kl_weight):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss * kl_weight


def train_vae(data, input_dim, latent_dim, kl_weight, epochs=50, batch_size=128, learning_rate=0.001):
    vae = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    data_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=batch_size,
                             shuffle=True)

    vae.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = loss_function(recon_x, x, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

    # 损失图
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='blue', label=f'Loss (kl_weight={kl_weight})')
    plt.title(f'Training Loss Over Epochs (kl_weight={kl_weight})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"loss_kl_weight_{kl_weight}.png")  # 保存图片
    plt.show()

    # 保存模型参数
    model_path = f"vae_model_kl_weight_{kl_weight}.pth"
    torch.save(vae.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")
    return vae, loss_history


def evaluate_on_new_data(vae, new_data, output_path, kl_weight):
    vae.eval()
    with torch.no_grad():
        x = torch.tensor(new_data, dtype=torch.float32)
        recon_x, mu, logvar = vae(x)
        loss = loss_function(recon_x, x, mu, logvar, kl_weight).item()

        print(f"Loss on new data: {loss}")

        # 保存结果
        with open(output_path, "w") as file:
            file.write(f"Loss on new data: {loss}\n")
            file.write("\nOriginal Data:\n")
            np.savetxt(file, new_data, fmt="%.6f")
            file.write("\nReconstructed Data:\n")
            np.savetxt(file, recon_x.numpy(), fmt="%.6f")
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # 加载训练数据
    path_train = "D:\\2024-12寒假\\p1"  # 修改为训练数据路径
    data_train = load_data(path_train, "data_sampled.pkl")
    data_train = data_train / np.max(data_train)  # 归一化到[0, 1]区间

    # 加载新数据
    path_test = "D:\\2024-12寒假\\p1"  # 修改为测试数据路径
    data_test = load_data(path_test, "sdata.pkl")
    data_test = data_test / np.max(data_test)  # 归一化到[0, 1]区间

    # 超参数搜索
    # kl_weights = [0.01, 0.1, 1, 2]
    # 权重范围
    kl_weights = [0.001, 0.01, 0.1, 0.5, 1, 2, 5]
    input_dim = 409
    latent_dim = 64  # 降维到64

    for kl_weight in kl_weights:
        print(f"\nTraining with kl_weight={kl_weight}")
        vae, loss_history = train_vae(data_train, input_dim, latent_dim, kl_weight)

        # 在新数据上评估
        output_path_test = f"evaluation_output_kl_weight_{kl_weight}.txt"
        evaluate_on_new_data(vae, data_test, output_path_test, kl_weight)
