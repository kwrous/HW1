import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNN
import os

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, 
                 learning_rate=1e-3, reg=1e-5, batch_size=64, 
                 num_epochs=10, lr_decay=0.95, verbose=True):
        """
        训练器初始化（添加了可视化相关参数）
        参数:
            model: 神经网络模型
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            learning_rate: 学习率
            reg: L2正则化强度
            batch_size: 批量大小
            num_epochs: 训练轮数
            lr_decay: 学习率衰减因子
            verbose: 是否打印训练信息
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.learning_rate = learning_rate
        self.reg = reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_decay = lr_decay
        self.verbose = verbose
        self.best_val_acc = 0
        self.best_params = {}
        
        # 新增记录容器
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def compute_loss(self, X, y):
        """计算损失（保持不变）"""
        num_samples = X.shape[0]
        scores = self.model.forward(X)
        correct_log_probs = -np.log(scores[np.arange(num_samples), y])
        data_loss = np.sum(correct_log_probs) / num_samples
        reg_loss = 0.5 * self.reg * (np.sum(self.model.params['W1']**2) + 
                                     np.sum(self.model.params['W2']**2) + 
                                     np.sum(self.model.params['W3']**2))
        return data_loss + reg_loss
    
    def compute_gradients(self, X, y):
        """计算梯度（保持不变）"""
        num_samples = X.shape[0]
        scores = self.model.forward(X)
        dscores = scores
        dscores[np.arange(num_samples), y] -= 1
        dscores /= num_samples
        self.model.backward(dscores)
        for param in ['W1', 'W2', 'W3']:
            self.model.grads[param] += self.reg * self.model.params[param]
    
    def update_parameters(self):
        """更新参数（保持不变）"""
        for param in self.model.params:
            self.model.params[param] -= self.learning_rate * self.model.grads[param]
    
    def check_accuracy(self, X, y):
        """计算准确率（保持不变）"""
        scores = self.model.forward(X)
        predicted_class = np.argmax(scores, axis=1)
        return np.mean(predicted_class == y)
    
    def _plot_progress(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='Train Loss', color='blue')
        plt.plot(self.val_loss_history, label='Val Loss', color='red', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label='Train Acc', color='green')
        plt.plot(self.val_acc_history, label='Val Acc', color='orange')
        plt.axhline(y=self.best_val_acc, color='gray', linestyle=':', 
                   label=f'Best Val Acc: {self.best_val_acc:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training/Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()
    
    def train(self):
        """训练过程（添加了损失计算和可视化）"""
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        
        for epoch in range(self.num_epochs):
            # 训练阶段
            epoch_loss = 0
            indices = np.random.permutation(num_train)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            for it in range(iterations_per_epoch):
                batch_idx = slice(it * self.batch_size, (it + 1) * self.batch_size)
                X_batch = X_shuffled[batch_idx]
                y_batch = y_shuffled[batch_idx]
                
                self.compute_gradients(X_batch, y_batch)
                self.update_parameters()
                epoch_loss += self.compute_loss(X_batch, y_batch)
            
            # 计算平均epoch损失
            epoch_loss /= iterations_per_epoch
            self.train_loss_history.append(epoch_loss)
            
            # 验证阶段
            val_loss = self.compute_loss(self.X_val, self.y_val)
            self.val_loss_history.append(val_loss)
            
            train_acc = self.check_accuracy(self.X_train, self.y_train)
            val_acc = self.check_accuracy(self.X_val, self.y_val)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {k: v.copy() for k, v in self.model.params.items()}
            
            # 学习率衰减
            self.learning_rate *= self.lr_decay
            
            # 打印信息
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}: "
                      f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            # 每5轮或最后保存一次图像
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                self._plot_progress()
        
        # 恢复最佳参数
        self.model.params = self.best_params
        self.save_best_model()
    
    def save_best_model(self, path='best_model.npz'):
        """保存最佳模型（保持不变）"""
        np.savez(path, **self.best_params)
        if self.verbose:
            print(f"Best model saved to {path} with val_acc={self.best_val_acc:.4f}")
    

