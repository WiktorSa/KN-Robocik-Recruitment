from utils import TrainAndSaveModel


if __name__ == '__main__':
    #TrainAndSaveModel(True, )
    train_data = np.load('preprocessed_data/train_data.npz')
    train_dataloader = get_data_loader(train_data['images'], train_data['gate_locations'],
                                       train_data['gate_coordinates'], 32, True)

    model = GateClassificationModel(3, 5)
    criterion = nn.CrossEntropyLoss()

    for x, y in train_dataloader:
        output = model.forward(x.float())
        loss = criterion(output, y.long())
        print(loss)
    print(sum(p.numel() for p in model.parameters()))
