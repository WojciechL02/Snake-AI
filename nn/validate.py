import torch
from torch.utils.data import DataLoader, random_split
from dataset import BCDataset
from model import MLP


def valid(model, device, loader, criterion):
    model.eval()
    correct = 0
    loss = 0
    n_samples = 0
    n_batches = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            n_samples += len(data)
            n_batches += 1

    acc = correct / n_samples
    loss /= n_batches
    print(f"\tVal: loss={loss:.3f}, acc={acc:.4f}")
    return loss, acc


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BCDataset("data")
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
                                    dataset,
                                    [train_size, val_size, test_size],
                                    generator=torch.Generator().manual_seed(42)
                                )
    test_loader = DataLoader(test_set, batch_size=64)

    _path_to_trained = "18_0.03_5_128_.pth"
    model_info = _path_to_trained.split("_")
    n_attrs = int(model_info[0])
    lr = float(model_info[1])
    lenght = int(model_info[2])
    width = int(model_info[3])

    model = MLP(input_size=n_attrs, output_size=4, hidden_layers=lenght, n_neurons=width)
    model.load_state_dict(torch.load(_path_to_trained))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    valid(model, device, test_loader, criterion)


if __name__ == "__main__":
    main()
