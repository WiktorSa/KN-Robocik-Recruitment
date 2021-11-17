import argparse
from preprocessing import preprocess_data


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dir', default='data', help='Directory with gate images', type=str, required=False)
    parser.add_argument('-ts', default=0.7, help='Size of train data', type=float, required=False)
    parser.add_argument('-vs', default=0.2, help='Size of validation data', type=float, required=False)
    parser.add_argument('--augmentation', dest='augmentate', help='Augmentate data', action='store_true')
    parser.add_argument('--no-augmentation', dest='augmentate', help='Do not augmentate data', action='store_false')
    parser.set_defaults(augmentate=False)
    parser.add_argument('-save_dir', default='preprocessed_data',
                        help='Directory where preprocessed data should be saved', type=str, required=False)
    parser.add_argument('-seed', default=1001, help='Seed', type=int, required=False)
    return parser


if __name__ == '__main__':
    data_parser = create_parser()
    args = data_parser.parse_args()
    preprocess_data(args.dir, args.ts, args.vs, args.augmentation, args.save_dir, args.seed)
