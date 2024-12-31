# @Author : LiZhongzheng
# 开发时间  ：2024-12-30 13:10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


# 数据加载函数
def load_data(path, filename="data_sampled.pkl"):
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
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss


def train_vae(data, input_dim, latent_dim, epochs=50, batch_size=128, learning_rate=0.001):
    vae = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    data_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=batch_size,
                             shuffle=True)

    # vae.train()
    # for epoch in range(epochs):
    #     total_loss = 0
    #     for batch in data_loader:
    #         x = batch[0]
    #         optimizer.zero_grad()
    #         recon_x, mu, logvar = vae(x)
    #         loss = loss_function(recon_x, x, mu, logvar)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

    vae.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

    # 损失图
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='blue', label='Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_picture02.png")  # 保存图片
    plt.show()

    return vae


if __name__ == "__main__":
    path = "D:\\2024-12寒假\\p1"  # 需要修改这里为自己的实际地址！！！！！！！！！！！！！！！！！！！！
    data = load_data(path)

    data = data / np.max(data)  # 归一化到[0, 1]区间

    # 训练VAE
    input_dim = 409
    latent_dim = 64  # 降维到64
    vae = train_vae(data, input_dim, latent_dim)

    vae.eval()
    with torch.no_grad():
        sample_data = torch.tensor(data[:5], dtype=torch.float32)  # 测试5条数据
        reconstructed, _, _ = vae(sample_data)

        print("Original Data:", sample_data)
        print("Reconstructed Data:", reconstructed)

        output_path = "D:\\2024-12寒假\\p1\\data_sampled_output.txt"  # 需要修改这里为自己的实际地址！！！！！！！！！！！！！！！！！！！！
        with open(output_path, "w") as file:
            file.write("Original Data:\n")
            np.savetxt(file, sample_data.numpy(), fmt="%.6f")
            file.write("\nReconstructed Data:\n")
            np.savetxt(file, reconstructed.numpy(), fmt="%.6f")

        print(f"Data saved to {output_path}")
