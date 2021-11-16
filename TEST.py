from model import GateClassificationModel, GateRegressionModel

if __name__ == '__main__':
    model = GateRegressionModel(3, 4)
    print(sum(p.numel() for p in model.parameters()))
