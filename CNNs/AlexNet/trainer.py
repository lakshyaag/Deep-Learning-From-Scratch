from config import Config
import mlflow
import torch
from data import prepare_data_loaders
from model import AlexNet
from rich import print
from torch import nn
from tqdm import tqdm

config = Config(N_EPOCHS=30)

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.enable_system_metrics_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model: AlexNet):
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

        for epoch in tqdm(range(config.N_EPOCHS)):
            # ---------- Training ----------
            train_loss = 0

            for batch, (inputs, labels) in enumerate(train_loader):
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
            model.eval()
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels)

                    total += labels.size(0)
                    correct += (labels == outputs.argmax(dim=-1)).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / total

            scheduler.step(val_loss)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "best_model")

        mlflow.pytorch.log_model(model, "final_model")
        mlflow.pytorch.save_model(model, "models")

    return run_id, val_loss, val_acc


def evaluate_model(model: AlexNet, run_id=None):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    with mlflow.start_run(run_id=run_id):
        with torch.inference_mode():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                test_loss += criterion(outputs, labels)

                total += labels.size(0)
                correct += (labels == outputs.argmax(dim=-1)).sum().item()

            test_loss /= len(test_loader)
            test_acc = correct / total

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)

    return test_loss, test_acc


def predict(model: AlexNet, data: torch.Tensor, true_labels: list[str]):
    model.eval()
    with torch.inference_mode():
        sample_test_images, sample_test_labels = data

        prediction = model(sample_test_images.to(device))
        predicted_labels = prediction.argmax(dim=-1).cpu().numpy()
        actual_labels = sample_test_labels.cpu().numpy()

        print(f"{'Index':<10}{'Predicted':<10}{'Actual':<10}")
        for i, (pred, actual) in enumerate(zip(predicted_labels, actual_labels)):
            print(f"{i:<10}{true_labels[pred]:<10}{true_labels[actual]:<10}", end="\n")


if __name__ == "__main__":
    train_loader, val_loader, test_loader, true_labels = prepare_data_loaders(
        config.BATCH_SIZE
    )

    model = AlexNet(n_classes=config.N_CLASSES, debug=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")

    run_id, val_loss, val_acc = train_model(model)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    test_loss, test_acc = evaluate_model(model, run_id)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    predict(model, next(iter(test_loader)), true_labels)
