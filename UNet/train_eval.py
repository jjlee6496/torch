import torch
import wandb
from data_loader import get_data_loaders
from UNet import UNet
from losses import CustomLoss
from optimizer import get_optimizer

def train(model, train_loader, criterion, optimizer, device, epoch, log_interval=1):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")
            wandb.log({"epoch": epoch+1, "batch_idx": batch_idx, "train_loss": loss.item()})


def main():
    data_root = ''
    batch_size = 32
    num_classes = 2
    learning_rate = 1e-3
    epochs = 10
    optimizer_type = 'sgd'
    momentum = 0.9
    log_interval = 10
    
    train_loader = get_data_loaders(data_root, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(num_classes).to(device)
    criterion = CustomLoss().to(device)
    optimizer = get_optimizer(model, optimizer_type, learning_rate, momentum)
    
    # Initialize wandb
    wandb.init(project='project1', config={"batch_size": batch_size, "learning_rate": learning_rate})
    
    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, log_interval)
    
    # Finish wandb logging
    wandb.finish()
    
if __name__ == '__main__':
    main()