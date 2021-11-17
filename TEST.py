from model import GateRegressionModel, GateClassificationModel

if __name__ == '__main__':
    model = GateClassificationModel(3, 4)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))