import torch
from tqdm import tqdm
from EarlyStopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter


# FINISH implementing classification
def train(model, train_dataloader, val_dataloader, criterion, optimizer, device: str, no_epochs: int,
          early_stopping: EarlyStopping, is_classification_task: bool) -> None:
    """
    Train a model and validate it's performance
    :param model: model to train on
    :param train_dataloader: training DataLoader
    :param val_dataloader: validation DataLoader
    :param criterion: criterion
    :param optimizer: optimizer
    :param device: the device to use in calculations. Either 'cpu' or 'cuda'
    :param no_epochs: the number of epochs to train the model
    :param early_stopping: early_stopping algorithm
    :param is_classification_task: are we training classification or regression model
    """

    writer = SummaryWriter()
    for epoch in range(no_epochs):
        # Training
        model.train()
        train_loss = 0

        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Train epoch {epoch + 1}')
        for i, (x, y) in enumerate(train_bar, 1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix_str(f'Train loss: {train_loss/i:.6f}')

        writer.add_scalar("Train loss", train_loss/len(train_dataloader), epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_dataloader, total=len(val_dataloader), desc=f'Val epoch {epoch + 1}')
        for i, (x, y) in enumerate(val_bar, 1):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)

            val_loss += loss.item()
            val_bar.set_postfix_str(f'Val loss: {val_loss/i:.6f}')

        writer.add_scalar("Val loss", val_loss / len(val_dataloader), epoch)

        early_stopping(val_loss / len(val_dataloader), model)
        if early_stopping.early_stop:
            break

    writer.flush()
    writer.close()
