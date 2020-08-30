import argparse

def main(args):
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some physics..')
    parser.add_argument('-n', '--n_gridpoints', type=int,
                        help='an integer')
    parser.add_argument('-o', '--output', type=str
                        help='the output file')

    args = parser.parse_args()

    main(args)
