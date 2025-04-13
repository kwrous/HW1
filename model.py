import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        初始化三层神经网络
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            activation: 激活函数类型 ('relu' 或 'sigmoid')
        """
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.params['b3'] = np.zeros(output_size)
        
        self.activation = activation
        self.grads = {}
        self.cache = {}
    
    def forward(self, X):
        """
        前向传播
        参数:
            X: 输入数据 (N, D)
        返回:
            out: 输出 (N, C)
        """
        # 第一层
        W1, b1 = self.params['W1'], self.params['b1']
        z1 = np.dot(X, W1) + b1
        if self.activation == 'relu':
            a1 = np.maximum(0, z1)
        elif self.activation == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))
        else:
            raise ValueError("激活函数必须是 'relu' 或 'sigmoid'")
        
        # 第二层
        W2, b2 = self.params['W2'], self.params['b2']
        z2 = np.dot(a1, W2) + b2
        if self.activation == 'relu':
            a2 = np.maximum(0, z2)
        else:
            a2 = 1 / (1 + np.exp(-z2))
        
        # 第三层 (输出层)
        W3, b3 = self.params['W3'], self.params['b3']
        z3 = np.dot(a2, W3) + b3
        out = self.softmax(z3)
        
        # 缓存中间结果用于反向传播
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3}
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        参数:
            dout: 损失函数对输出的梯度 (N, C)
        返回:
            无 (梯度存储在 self.grads 中)
        """
        X = self.cache['X']
        a1, a2 = self.cache['a1'], self.cache['a2']
        z1, z2 = self.cache['z1'], self.cache['z2']
        
        # 第三层梯度
        dz3 = dout
        self.grads['W3'] = np.dot(a2.T, dz3)
        self.grads['b3'] = np.sum(dz3, axis=0)
        
        # 第二层梯度
        da2 = np.dot(dz3, self.params['W3'].T)
        if self.activation == 'relu':
            dz2 = da2 * (a2 > 0)
        else:
            dz2 = da2 * a2 * (1 - a2)
        self.grads['W2'] = np.dot(a1.T, dz2)
        self.grads['b2'] = np.sum(dz2, axis=0)
        
        # 第一层梯度
        da1 = np.dot(dz2, self.params['W2'].T)
        if self.activation == 'relu':
            dz1 = da1 * (a1 > 0)
        else:
            dz1 = da1 * a1 * (1 - a1)
        self.grads['W1'] = np.dot(X.T, dz1)
        self.grads['b1'] = np.sum(dz1, axis=0)
    
    def softmax(self, x):
        """
        softmax函数
        参数:
            x: 输入 (N, C)
        返回:
            softmax输出 (N, C)
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def save_weights(self, path):
        """
        保存模型权重
        参数:
            path: 保存路径
        """
        np.savez(path, **self.params)
    
    def load_weights(self, path):
        """
        加载模型权重
        参数:
            path: 权重文件路径
        """
        weights = np.load(path)
        for key in self.params:
            self.params[key] = weights[key]