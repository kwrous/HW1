import numpy as np
from sklearn.model_selection import train_test_split
from model import ThreeLayerNN
from train import Trainer
from test import Tester
from hyperparam_search import HyperparameterSearch
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms

def load_and_preprocess_data():
    """使用torchvision加载CIFAR-10数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载训练和测试集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 转换为numpy数组
    X_train = trainset.data
    y_train = np.array(trainset.targets)
    X_test = testset.data
    y_test = np.array(testset.targets)
    
    # 归一化并展平
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    # 超参数搜索
    print("Starting hyperparameter search...")
    search = HyperparameterSearch(X_train, y_train, X_val, y_val)
    search.search(
        hidden_sizes=[128, 256, 512],
        learning_rates=[1e-3, 5e-4, 1e-4],
        reg_strengths=[1e-5, 5e-5],
        activations=['relu', 'sigmoid'],
        num_epochs=10,
        batch_size=64
    )
    search.save_results()
    
    # 获取最佳参数
    best_params = search.get_best_params()
    print("\nBest parameters found:")
    print(f"hidden_size: {best_params['hidden_size']}")
    print(f"learning_rate: {best_params['learning_rate']}")
    print(f"reg: {best_params['reg']}")
    print(f"activation: {best_params['activation']}")
    print(f"validation accuracy: {best_params['val_acc']:.4f}")
    
    # 使用最佳参数训练最终模型
    print("\nTraining final model with best parameters...")
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    model = ThreeLayerNN(
        input_size, best_params['hidden_size'], output_size, 
        activation=best_params['activation'])
    
    trainer = Trainer(
        model, X_train, y_train, X_val, y_val,
        learning_rate=best_params['learning_rate'],
        reg=best_params['reg'],
        batch_size=64,
        num_epochs=20,
        lr_decay=0.95
    )
    trainer.train()
    trainer.save_best_model('final_model.npz')
    
    # 测试最终模型
    tester = Tester(model, X_test, y_test)
    test_acc = tester.evaluate()
    print(f"\nFinal test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()