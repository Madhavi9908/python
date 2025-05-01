
# train.py
import torch
import torch.optim as optim
from src.config import Config
from src.data_loader import get_data_loaders
from src.model import MNISTNet
from src.trainer import Trainer
from src.utils import save_model, plot_training_history

def main():
    config = Config()
    train_loader, test_loader = get_data_loaders(config)
    
    model = MNISTNet().to(config.DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    
    trainer = Trainer(model, optimizer, config.DEVICE)
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f'\nEpoch {epoch}/{config.NUM_EPOCHS}')
        
        train_loss, train_acc = trainer.train_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    save_model(model, config.MODEL_SAVE_PATH)
    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)

if __name__ == '__main__':
    main()