from tensorflow.keras.datasets import mnist, cifar10

# Load MNIST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print(f"MNIST Dataset: Train - {X_train_mnist.shape}, Test - {X_test_mnist.shape}")

# Load CIFAR-10
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
print(f"CIFAR-10 Dataset: Train - {X_train_cifar.shape}, Test - {X_test_cifar.shape}")