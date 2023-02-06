from validate import valid
from dataset import BCDataset
from model import MLP

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    correct = 0
    total_loss = 0
    n_samples = 0
    n_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_samples += len(data)
        n_batches += 1

    if epoch == 1:
        layers_norm = torch.zeros(size=(1, 32))
        j = 0
        for i in range(0, len(model.layers), 2):
            params = model.layers[i].weight.grad
            norm = torch.norm(input=params)
            layers_norm[0, j] = norm
            j += 1;
        print(layers_norm)

    acc = correct / n_samples
    loss = total_loss / n_batches
    print(f"Epoch {epoch} | Train: loss={loss:.3f}, acc={acc:.4f}")
    return loss, acc


def train(model, device, loaders, writer, criterion, optimizer, n_epochs, path_to_save_model):
    train_loader, val_loader = loaders["train"], loaders["val"]

    best_vacc = 0
    for epoch in range(1, n_epochs+1):
        loss, acc = train_one_epoch(model, device, train_loader, criterion, optimizer, epoch)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)

        val_loss, val_acc = valid(model, device, val_loader, criterion)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_acc > best_vacc:
            best_vacc = val_acc
            torch.save(model.state_dict(), path_to_save_model)


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    N_ATTRS = 18
    N_CATEGORIES = 4  # UP/RIGHT/DOWN/LEFT

    N_HIDDEN_LAYERS = 30
    N_NEURONS = 64
    BATCH_SIZE = 64
    LEARNING_RATE = 0.03
    EPOCHS = 5

    dataset = BCDataset("data")
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
                                    dataset,
                                    [train_size, val_size, test_size],
                                    generator=torch.Generator().manual_seed(42)
                                )

    loaders = {}
    loaders["train"] = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    loaders["val"] = DataLoader(val_set, batch_size=BATCH_SIZE)
    loaders["test"] = DataLoader(test_set, batch_size=BATCH_SIZE)

    writer = SummaryWriter()
    model = MLP(input_size=N_ATTRS, output_size=N_CATEGORIES, hidden_layers=N_HIDDEN_LAYERS, n_neurons=N_NEURONS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # e.q.: 15_4_3_64.pth
    path_to_save_model = f"{N_ATTRS}_{LEARNING_RATE}_{N_HIDDEN_LAYERS}_{N_NEURONS}_.pth"
    train(model, device, loaders, writer, criterion, optimizer, EPOCHS, path_to_save_model)


if __name__ == "__main__":
    main()
