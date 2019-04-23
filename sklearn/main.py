from __future__ import print_function
from wine_quality.train import Trainer
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", required=True)
    parser.add_argument("--data_path", dest="data_path", help="data_path", required=True)
    parser.add_argument("--alpha", dest="alpha", help="alpha", default=0.1, type=float )
    parser.add_argument("--l1_ratio", dest="l1_ratio", help="l1_ratio", default=0.1, type=float )
    parser.add_argument("--run_origin", dest="run_origin", help="run_origin", default="none")
    args = parser.parse_args()
    trainer = Trainer(args.experiment_name, args.data_path,args.run_origin)
    trainer.train(args.alpha, args.l1_ratio)
