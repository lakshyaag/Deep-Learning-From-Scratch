from config import Config
import mlflow
import torch
from data import prepare_data_loaders
from model import ResNet
from rich import print
from torch import nn
from tqdm import tqdm

config = Config(
    BATCH_SIZE=64,
    N_EPOCHS=30,
    N_CLASSES=100,
    WEIGHT_DECAY=0.0001,
    N_BLOCKS=9,
    LEARNING_RATE=0.1,
)

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.enable_system_metrics_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model: ResNet):
    mlflow.set_experiment(experiment_name=config.EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # Log hyperparameters
        mlflow.log_params(
            {
                **config.__dict__,
                "Optimizer": optimizer.__class__.__name__,
                "Loss Function": criterion.__class__.__name__,
            }
        )

        best_val_loss = float("inf")

        for epoch in tqdm(range(config.N_EPOCHS), desc="Epochs"):
            # ---------- Training ----------
            train_loss = 0

            for batch, (inputs, labels) in enumerate(
                tqdm(train_loader, desc="Training")
            ):
                model.train()

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= len(train_loader)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # ---------- Validation ----------

            val_loss = 0
            correct = 0
            total = 0
            label_in_top_5 = 0

            model.eval()
            with torch.inference_mode():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels)

                    top5_preds = outputs.topk(k=5, dim=-1).indices

                    total += labels.size(0)
                    correct += (labels == outputs.argmax(dim=-1)).sum().item()
                    label_in_top_5 += (
                        (labels.view(-1, 1) == top5_preds).any(dim=-1).sum().item()
                    )

                val_loss /= len(val_loader)
                val_acc = correct / total
                val_top5_acc = label_in_top_5 / total

            scheduler.step(val_loss)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_top5_acc", val_top5_acc, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "best_model")

        mlflow.pytorch.log_model(model, "final_model")
        mlflow.pytorch.save_model(model, "models")

    return run_id, val_loss, val_acc


def evaluate_model(model: ResNet, run_id=None):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    label_in_top_5 = 0
    with mlflow.start_run(run_id=run_id):
        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                test_loss += criterion(outputs, labels)

                top5_preds = outputs.topk(k=5, dim=-1).indices

                total += labels.size(0)
                correct += (labels == outputs.argmax(dim=-1)).sum().item()
                label_in_top_5 += (
                    (labels.view(-1, 1) == top5_preds).any(dim=-1).sum().item()
                )

            test_loss /= len(test_loader)
            test_acc = correct / total
            test_top5_acc = label_in_top_5 / total

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_top5_acc", test_top5_acc)

    return test_loss, test_acc


def predict(model: ResNet, data: torch.Tensor, true_labels: list[str]):
    model.eval()
    with torch.inference_mode():
        sample_test_images, sample_test_labels = data

        prediction = model(sample_test_images.to(device))
        predicted_labels = prediction.argmax(dim=-1).cpu().numpy()
        actual_labels = sample_test_labels.cpu().numpy()

        print(f"{'Index':<15}{'Predicted':<15}{'Actual':<15}")
        for i, (pred, actual) in enumerate(zip(predicted_labels, actual_labels)):
            print(f"{i:<15}{true_labels[pred]:<15}{true_labels[actual]:<15}", end="\n")


if __name__ == "__main__":
    train_loader, val_loader, test_loader, true_labels = prepare_data_loaders(
        config.BATCH_SIZE, train_split=0.7
    )

    model = ResNet(
        n_classes=config.N_CLASSES, n_blocks=config.N_BLOCKS, debug=False
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.1
    )

    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")

    run_id, val_loss, val_acc = train_model(model)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    test_loss, test_acc = evaluate_model(model, run_id)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    predict(model, next(iter(test_loader)), true_labels)
