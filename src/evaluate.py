import torch
from sklearn.metrics import confusion_matrix

def evaluate_model(model, dataloader, loss_fn, device, classes):

    model.eval()
    total, correct, running_loss = 0, 0, 0

    class_correct = [0 for _ in classes]
    class_total = [0 for _ in classes]

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

            for i in range(labels.size(0)):
                label = labels[i]
                pred = preds[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    accuracy = 100 * correct / total

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Loss: {running_loss / len(dataloader)}")

    print("\nAccuracy per class:")
    for i, cls in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{cls}: {acc:.2f}%")

    cm = confusion_matrix(labels_all, preds_all)

    return accuracy, cm