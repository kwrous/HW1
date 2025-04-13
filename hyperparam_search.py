import numpy as np
import itertools
import matplotlib.pyplot as plt
from model import ThreeLayerNN
from train import Trainer

class HyperparameterSearch:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.results = []
        self.best_model = None  
    
    def _visualize_parameters(self, model, params_str):
        """可视化网络参数"""
        plt.figure(figsize=(15, 10))
        
        # 1. 第一层权重分布
        plt.subplot(2, 2, 1)
        w1 = model.params['W1']
        plt.hist(w1.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title(f'Layer1 Weights Distribution\n{params_str}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        
        # 2. 第二层权重热力图
        plt.subplot(2, 2, 2)
        w2 = model.params['W2']
        # 随机选取100个神经元展示
        sample_neurons = np.random.choice(w2.shape[1], 100, replace=False)
        plt.imshow(w2[:, sample_neurons], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Layer2 Weights Heatmap (100 random neurons)')
        plt.xlabel('Neuron Index')
        plt.ylabel('Input Dimension')
        
        # 3. 权重绝对值分布比较
        plt.subplot(2, 2, 3)
        abs_w1 = np.abs(w1.flatten())
        abs_w2 = np.abs(w2.flatten())
        plt.hist(abs_w1, bins=50, alpha=0.5, label='Layer1', color='blue')
        plt.hist(abs_w2, bins=50, alpha=0.5, label='Layer2', color='green')
        plt.title('Absolute Weights Distribution')
        plt.legend()
        plt.xlabel('Absolute Weight Value')
        plt.ylabel('Frequency')
        
        # 4. 输出层权重统计
        plt.subplot(2, 2, 4)
        w3 = model.params['W3']
        class_weights = np.linalg.norm(w3, axis=0)
        plt.bar(range(len(class_weights)), class_weights)
        plt.title('Output Layer Weights Norm by Class')
        plt.xlabel('Class Index')
        plt.ylabel('Weight Norm')
        
        plt.tight_layout()
        plt.savefig(f'weights_visualization_{params_str.replace(" ", "_")}.png')
        plt.close()
    
    def search(self, hidden_sizes, learning_rates, reg_strengths, 
               activations=['relu'], num_epochs=10, batch_size=64):
        param_combinations = itertools.product(
            hidden_sizes, learning_rates, reg_strengths, activations)
        
        for hidden_size, lr, reg, activation in param_combinations:
            params_str = f"hs{hidden_size}_lr{lr}_reg{reg}_{activation}"
            print(f"\nTesting params: {params_str}")
            
            # 初始化模型
            input_size = self.X_train.shape[1]
            output_size = len(np.unique(self.y_train))
            model = ThreeLayerNN(input_size, hidden_size, output_size, activation)
            
            # 训练模型
            trainer = Trainer(
                model, self.X_train, self.y_train, self.X_val, self.y_val,
                learning_rate=lr, reg=reg, batch_size=batch_size, 
                num_epochs=num_epochs, verbose=False)
            trainer.train()
            
            # 参数可视化
            self._visualize_parameters(model, params_str)
            
            # 记录结果
            result = {
                'hidden_size': hidden_size,
                'learning_rate': lr,
                'reg': reg,
                'activation': activation,
                'val_acc': trainer.best_val_acc,
                'train_acc': trainer.train_acc_history[-1],
                'params_str': params_str
            }
            self.results.append(result)
            
            # 更新最佳模型
            if self.best_model is None or result['val_acc'] > self.best_model['val_acc']:
                self.best_model = result.copy()
                self.best_model['model'] = model
            
            print(f"Train acc: {result['train_acc']:.4f}, Val acc: {result['val_acc']:.4f}")
        
        # 可视化最佳模型参数
        if self.best_model:
            print("\nVisualizing best model parameters...")
            best_params_str = self.best_model['params_str']
            self._visualize_parameters(self.best_model['model'], 
                                      f"BEST_{best_params_str}_acc{self.best_model['val_acc']:.4f}")
    
    def get_best_params(self):
        if not self.results:
            return None
        return max(self.results, key=lambda x: x['val_acc'])
    
    def save_results(self, path='hyperparam_results.txt'):
        with open(path, 'w') as f:
            # 写入所有结果
            for result in sorted(self.results, key=lambda x: -x['val_acc']):
                f.write(f"{result['params_str']}: "
                       f"train_acc={result['train_acc']:.4f}, "
                       f"val_acc={result['val_acc']:.4f}\n")
            
            # 写入最佳结果
            best = self.get_best_params()
            if best:
                f.write("\n=== BEST PARAMETERS ===\n")
                f.write(f"Hidden size: {best['hidden_size']}\n")
                f.write(f"Learning rate: {best['learning_rate']}\n")
                f.write(f"Regularization: {best['reg']}\n")
                f.write(f"Activation: {best['activation']}\n")
                f.write(f"Validation accuracy: {best['val_acc']:.4f}\n")