import nn
import pandas as pd


class minst(nn.nn):
    def __init__(self, n_input, n_interm, n_output):
        nn.nn.__init__(self, n_input, n_interm, n_output)

    def gen_inputs_wrapper(self, fp):
        df = pd.read_csv(fp)
        print(df.label.unique)
        print(df.mean())


def demo():
    fp = "./mnist/train.csv"
    n = minst(3, 3, 2)
    n.gen_inputs_wrapper(fp)

if __name__ == '__main__':
    demo()