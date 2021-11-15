import argparse
from utils import train_and_save_model


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-task', help='Train classification task or regression task',
                        choices=['classification', 'regression'], type=str, required=True)
    parser.add_argument('-dir', default='preprocessed_data', help='Directory with preprocessed data', type=str,
                        required=False)
    parser.add_argument('-bs', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('-lr', default=1e-3, help='Learning rate', type=float, required=False)
    parser.add_argument('-ep', default=100, help='Number of epochs', type=int, required=False)
    parser.add_argument('-save_dir', default='model_weights', help='Directory where model weights should be saved',
                        type=str, required=False)
    parser.add_argument('-seed', default=1001, help='Seed', type=int, required=False)
    return parser


if __name__ == '__main__':
    data_parser = create_parser()
    args = data_parser.parse_args()

    if args.task == 'classification':
        train_and_save_model(True, args.dir, args.bs, args.lr, args.ep, args.save_dir, args.seed)

    elif args.task == 'regression':
        train_and_save_model(False, args.dir, args.bs, args.lr, args.ep, args.save_dir, args.seed)

    else:
        data_parser.error("Not recognized task. Type classification or regression")
