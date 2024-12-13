import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torchmetrics
from torchvision.models import VGG19_Weights, VGG16_Weights

class CIFAR10Module(pl.LightningModule):
    def __init__(self, model_name='vgg19', dropout_p=0.3, freeze_layers=20, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Model initialization
        if model_name == 'vgg19':
            self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg16':
            self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Add dropout
        classifier = list(self.model.classifier.children())
        classifier.insert(1, nn.Dropout(p=dropout_p))
        self.model.classifier = nn.Sequential(*classifier)
        
        # Freeze layers
        for i, param in enumerate(self.model.features.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
        
        # Modify classifier for CIFAR10
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 10)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# Data module
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_train_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform_train
            )
            
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, 
                [train_size, val_size]
            )
            self.val_dataset.dataset.transform = self.transform_test

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Training script
def train_model():
    # Initialize data module
    data_module = CIFAR10DataModule()
    
    # Initialize model
    model = CIFAR10Module()
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='cifar10-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator='auto',
        devices=1,
        callbacks=[early_stopping, checkpoint_callback]
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)

if __name__ == '__main__':
    train_model()