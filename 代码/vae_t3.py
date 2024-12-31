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
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss


def train_vae(data, input_dim, latent_dim, epochs=50, batch_size=128, learning_rate=0.001):
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

    # save
    # 保存模型参数
    model_path = "D:\\2024-12寒假\\p1\\vae_model.pth"  # 修改为模型保存路径
    torch.save(vae.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")
    return vae


def evaluate_on_new_data(vae, new_data, output_path):
    vae.eval()
    with torch.no_grad():
        x = torch.tensor(new_data, dtype=torch.float32)
        recon_x, mu, logvar = vae(x)
        loss = loss_function(recon_x, x, mu, logvar).item()

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

    # 训练VAE
    input_dim = 409
    latent_dim = 64  # 降维到64
    vae = train_vae(data_train, input_dim, latent_dim)

    # 加载新数据
    path_test = "D:\\2024-12寒假\\p1"  # 修改为测试数据路径
    data_test = load_data(path_test, "sdata.pkl")
    data_test = data_test / np.max(data_test)  # 归一化到[0, 1]区间

    # 在新数据上计算损失并保存结果
    output_path_test = "D:\\2024-12寒假\\p1\\evaluation_output.txt"  # 修改为输出路径
    evaluate_on_new_data(vae, data_test, output_path_test)

    # 重新加载保存的模型并计算新的损失
    model_path = "D:\\2024-12寒假\\p1\\vae_model.pth"  # 修改为模型保存路径
    vae_loaded = VAE(input_dim, latent_dim)
    vae_loaded.load_state_dict(torch.load(model_path, weights_only=True))  # 加载保存的模型权重

    # 使用加载的模型进行评估
    output_path_test_loaded = "D:\\2024-12寒假\\p1\\evaluation_output_loaded.txt"  # 输出路径
    evaluate_on_new_data(vae_loaded, data_test, output_path_test_loaded)
