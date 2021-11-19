import argparse
from visualisation import visualise_results


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dir_data', default='preprocessed_data', help='Directory with preprocessed data', type=str,
                        required=False)
    parser.add_argument('-dir_models', default='model_weights', help='Directory with model weights', type=str,
                        required=False)
    parser.add_argument('-width', default=800, help='At what width will images be shown', type=int, required=False)
    parser.add_argument('-height', default=500, help='At what height will images be shown', type=int, required=False)
    parser.add_argument('-seed', default=None, help='Seed', type=int, required=False)
    return parser


if __name__ == '__main__':
    data_parser = create_parser()
    args = data_parser.parse_args()
    visualise_results(args.dir_data, args.dir_models, args.width, args.height, args.seed)
