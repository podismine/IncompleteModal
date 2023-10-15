import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d', type=str, default="AD")
    parser.add_argument('--channel','-ch', type=int, default=32, help='channel number')
    parser.add_argument('--complete','-cm', type=int, default=5, help='complete data ratio')
    parser.add_argument('-t','--type', type=str, default='c',choices=['c','knn','adv','mixup','mean'], help='data completation type')

    parser.add_argument('--layer','-l', type=int, default=2, help='layer number')
    parser.add_argument('--ab','-a', type=int, default=0, help='layer number')
    parser.add_argument('--mm', type=int, default=3, help='layer number')
    parser.add_argument('--no', type=int, default=2, help='layer number')
    parser.add_argument('-g','--gru', type=int, default=1, help='layer number')
    parser.add_argument('--alpha', type=float, default=0.8, help='layer number')

    args = parser.parse_args()
    args.dataset = args.dataset.upper()

    return args