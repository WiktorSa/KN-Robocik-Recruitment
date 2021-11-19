import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_regression(model, train_dataloader, val_dataloader, criterion, optimizer, device: str, no_epochs: int,
                     early_stopping) -> None:
    """
    Train a regression model and validate it's performance

    :param model: model to train on
    :param train_dataloader: training DataLoader
    :param val_dataloader: validation DataLoader
    :param criterion: criterion
    :param optimizer: optimizer
    :param device: the device to use in calculations. Either 'cpu' or 'cuda'
    :param no_epochs: the number of epochs to train the model
    :param early_stopping: early stopping algorithm
    """

    writer = SummaryWriter()
    for epoch in range(no_epochs):
        # Training
        model.train()
        train_loss = 0

        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Train epoch {epoch + 1}')
        for i, (x, y) in enumerate(train_bar, 1):
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix_str(f'Train loss: {train_loss / i:.2f}')

        writer.add_scalar("Train loss", train_loss / len(train_dataloader), epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_dataloader, total=len(val_dataloader), desc=f'Val epoch {epoch + 1}')
        for i, (x, y) in enumerate(val_bar, 1):
            x = x.float().to(device)
            y = y.float().to(device)

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)

            val_loss += loss.item()
            val_bar.set_postfix_str(f'Val loss: {val_loss / i:.2f}')

        writer.add_scalar("Val loss", val_loss / len(val_dataloader), epoch)

        if early_stopping(val_loss / len(val_dataloader), model):
            break

    writer.flush()
    writer.close()


def train_classification(model, train_dataloader, val_dataloader, criterion, optimizer, device: str, no_epochs: int,
                         early_stopping) -> None:
    """
    Train a classification model and validate it's performance

    :param model: model to train on
    :param train_dataloader: training DataLoader
    :param val_dataloader: validation DataLoader
    :param criterion: criterion
    :param optimizer: optimizer
    :param device: the device to use in calculations. Either 'cpu' or 'cuda'
    :param no_epochs: the number of epochs to train the model
    :param early_stopping: early stopping algorithm
    """

    writer = SummaryWriter()
    for epoch in range(no_epochs):
        # Training
        model.train()
        train_loss = 0
        train_inputs = 0
        train_correct = 0

        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Train epoch {epoch + 1}')
        for x, y in train_bar:
            x = x.float().to(device)
            y = y.long().to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_inputs += len(y)
            _, predicted_code = torch.max(y_pred, 1)
            train_correct += (predicted_code == y).sum().item()

            train_bar.set_postfix_str(f'Train loss: {train_loss / train_inputs:.4f}, '
                                      f'accuracy: {train_correct / train_inputs:.4f}')

        writer.add_scalar("Train loss", train_loss / train_inputs, epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_inputs = 0
        val_correct = 0

        val_bar = tqdm(val_dataloader, total=len(val_dataloader), desc=f'Val epoch {epoch + 1}')
        for i, (x, y) in enumerate(val_bar, 1):
            x = x.float().to(device)
            y = y.long().to(device)

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)

            val_loss += loss.item()
            val_inputs += len(y)
            _, predicted_code = torch.max(y_pred, 1)
            val_correct += (predicted_code == y).sum().item()

            val_bar.set_postfix_str(f'Val loss: {val_loss / val_inputs:.4f}, '
                                    f'accuracy: {val_correct / val_inputs:.4f}')

        writer.add_scalar("Val loss", val_loss / val_inputs, epoch)

        if early_stopping(val_loss / val_inputs, model):
            break

    writer.flush()
    writer.close()
