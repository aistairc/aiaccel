import argparse


class Arguments:
    def __new__(cls):
        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument('--config', '-c', type=str)
        p.add_argument('--resume', type=int, default=None)
        p.add_argument('--clean', nargs='?', const=True, default=False)
        p.add_argument('--nosave', nargs='?', const=True, default=False)
        p.add_argument('--fs', nargs='?', const=True, default=False)
        return vars(p.parse_args())
