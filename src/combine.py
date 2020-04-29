import os
import xlrd
import argparse

from pathlib import Path


class HParams:
    """
    Class that takes in all model training parameters.
    """

    def __init__(self):
        self.args = self.get_parser()

    def get_parser(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--folder', type=Path, default='', 
                                help='Directory that one wants to search for results')
        return arg_parser.parse_args()                          


def list_files(folder: Path):
    r = []
    for root, folders, files in os.walk(folder):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def get_gentexts(lst: list):
    gen_texts = []
    for f_name in lst:
        with open(f_name) as f:
            gen_text = f.read()
        gen_texts.append(gen_text)
    return gen_texts


def main(folder: Path):

    f_list = list_files(folder)
    gen_texts = get_gentexts(f_list)


if __name__ == '__main__':
    parameters = HParams().args
    main(args.folder)
