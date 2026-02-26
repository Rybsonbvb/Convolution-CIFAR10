import torch
from src.model import CIFAR10CNN
from src.dataset import get_dataloaders
from src.train import train
from src.evaluate import evaluate_model
from src.utils import plot_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, classes = get_dataloaders()

model = CIFAR10CNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

history = train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs=100,
    device=device,
)

torch.save(model.state_dict(), "models/best_model.pth")

plot_training(history)

evaluate_model(model, test_loader, loss_fn, device, classes)