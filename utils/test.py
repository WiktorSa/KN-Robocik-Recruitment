import torch
from tqdm import tqdm


def test_regression(model, test_dataloader, criterion, device: str) -> None:
    """
    Test a regression model

    :param model: model to test
    :param test_dataloader: test DataLoader
    :param criterion: criterion
    :param device: the device to use in calculations. Either 'cpu' or 'cuda'
    """

    model.eval()
    test_loss = 0
    test_bar = tqdm(test_dataloader, total=len(test_dataloader), desc=f'Test')
    for i, (x, y) in enumerate(test_bar, 1):
        x = x.float().to(device)
        y = y.float().to(device)

        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)

        test_loss += loss.item()
        test_bar.set_postfix_str(f'Test loss: {test_loss/i:.2f}')


def test_classification(model, test_dataloader, criterion, device: str) -> None:
    """
    Test a classification model

    :param model: model to test
    :param test_dataloader: test DataLoader
    :param criterion: criterion
    :param device: the device to use in calculations. Either 'cpu' or 'cuda'
    """

    model.eval()
    test_loss = 0
    test_inputs = 0
    test_correct = 0
    test_bar = tqdm(test_dataloader, total=len(test_dataloader), desc=f'Test')
    for x, y in test_bar:
        x = x.float().to(device)
        y = y.long().to(device)

        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)

        test_loss += loss.item()
        test_inputs += len(y)
        _, predicted_code = torch.max(y_pred, 1)
        test_correct += (predicted_code == y).sum().item()

        test_bar.set_postfix_str(f'Test loss: {test_loss/test_inputs:.4f} '
                                 f'accuracy: {test_correct / test_inputs:.4f}')
