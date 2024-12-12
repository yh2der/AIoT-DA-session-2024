import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms remain the same
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset loading
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_test

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def add_dropout(model, p=0.3):
    classifier = list(model.classifier.children())
    classifier.insert(1, nn.Dropout(p=0.3))
    model.classifier = nn.Sequential(*classifier)
    
    return model

def freeze_layers(model, num_layers_to_freeze):
    # 凍結特徵提取器中的層
    for i, param in enumerate(model.features.parameters()):
        if i < num_layers_to_freeze:
            param.requires_grad = False
            
    # 只保留最後一個線性層可訓練
    for i, param in enumerate(model.classifier.parameters()):
        if i < len(list(model.classifier.parameters())) - 2:  # 只保留最後一層可訓練
            param.requires_grad = False
            
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_train_loss': [],
        'epoch_val_loss': []
    }
    
    early_stopping = EarlyStopping(patience=3)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_train_losses = []
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_progress = tqdm(train_loader, desc=f"Training", leave=False)
        
        for inputs, labels in epoch_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            epoch_progress.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss.item())
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        epoch_val_loss = sum(val_losses) / len(val_losses)
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        
        # Store metrics
        history['train_loss'].extend(epoch_train_losses)
        history['epoch_train_loss'].append(epoch_train_loss)
        history['epoch_val_loss'].append(epoch_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Early stopping check
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return history

def plot_training_history(history, model_name):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch_train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['epoch_val_loss'], label='Validation Loss', linewidth=2)
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Load and modify VGG16
# vgg16 = models.vgg16(pretrained=True)
# vgg16 = add_dropout(vgg16)  # 增加dropout
# vgg16 = freeze_layers(vgg16, 20)  # 凍結前20層
# vgg16.classifier[-1] = nn.Linear(vgg16.classifier[-1].in_features, 10)
# vgg16 = vgg16.to(device)

# Load and modify VGG19
vgg19 = models.vgg19(pretrained=True)
vgg19 = add_dropout(vgg19)  # 增加更多的dropout
vgg19 = freeze_layers(vgg19, 20)  # 凍結更多層
vgg19.classifier[-1] = nn.Linear(vgg19.classifier[-1].in_features, 10)
vgg19 = vgg19.to(device)

# 使用SGD優化器
# optimizer_vgg16 = optim.SGD(vgg16.parameters(),
#                            lr=0.01,
#                            momentum=0.9,
#                            weight_decay=5e-4)

optimizer_vgg19 = optim.SGD(
    vgg19.parameters(),
    lr=0.01,  # 稍微提高學習率
    momentum=0.9,
    weight_decay=5e-4,  # 降低權重衰減
)

# 添加學習率調度器
# scheduler_vgg16 = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_vgg16,
#     mode='min',
#     factor=0.1,
#     patience=3,
#     verbose=True
# )

scheduler_vgg19 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_vgg19,
    mode='min',
    factor=0.1,
    patience=3,  # 增加耐心度
    verbose=True,
)

criterion = nn.CrossEntropyLoss()

# Train VGG16
# print("Training VGG16...")
# vgg16_history = train_model(
#     vgg16, 
#     train_loader, 
#     val_loader, 
#     criterion, 
#     optimizer_vgg16, 
#     scheduler_vgg16,
#     num_epochs=10
# )

# # Plot results
# plot_training_history(vgg16_history, 'VGG16')

# Train VGG19
print("Training VGG19...")
vgg19_history = train_model(
    vgg19, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer_vgg19, 
    scheduler_vgg19,
    num_epochs=15
)

# Plot results
plot_training_history(vgg19_history, 'VGG19')