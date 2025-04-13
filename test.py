import numpy as np
from model import ThreeLayerNN

class Tester:
    def __init__(self, model, X_test, y_test):
        """
        测试器初始化
        参数:
            model: 神经网络模型
            X_test: 测试数据
            y_test: 测试标签
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate(self):
        """
        评估模型在测试集上的性能
        返回:
            测试准确率
        """
        scores = self.model.forward(self.X_test)
        predicted_class = np.argmax(scores, axis=1)
        accuracy = np.mean(predicted_class == self.y_test)
        return accuracy
    
    def evaluate_with_loaded_model(self, model_path):
        """
        加载模型并评估
        参数:
            model_path: 模型权重路径
        返回:
            测试准确率
        """
        self.model.load_weights(model_path)
        return self.evaluate()