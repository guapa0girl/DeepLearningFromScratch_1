# dataset.mnist.py 내용
import torch
from torchvision import datasets

def load_mnist(flatten=True, normalize=False):
    # 1. data downdoad
    train_ds = datasets.MNIST(root='./data',train=True,download=True)
    test_ds = datasets.MNIST(root='./data',train=False,download=True)
    
    # 2. numpy 변환
    x_train = train_ds.data.numpy()
    t_train = test_ds.targets.numpy()
    x_test = test_ds.data.numpy()
    t_test = test_ds.targets.numpy()
    
    # 3. 정규화 (normalize=True일 경우 0~1 사이 값으로 변환)
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
    # 4. 평탄화 (flatten=True일 경우 1차원 배열로 변환)
    if flatten:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
        
    return (x_train, t_train), (x_test, t_test)
