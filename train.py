import argparse
from trainer import Trainer


def train(args):
    trainer = Trainer(args=args)
    trainer.train()  
    print('trained')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--iter-d', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.9)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--lambp', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--continue-training', action='store_true')
    parser.add_argument('--netd-checkpoint', type=str, default='')
    parser.add_argument('--netg-checkpoint', type=str, default='')
    parser.add_argument('--use-valset', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--ds-name', type=str, default='CUB2011')
    parser.add_argument('--ds-path', type=str, default='/')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--n-cls', type=int, default=200)
    parser.add_argument('--n-z', type=int, default=200)
    parser.add_argument('--n-feat', type=int, default=3584)
    parser.add_argument('--n-tfidf', type=int, default=11083)
    parser.add_argument('--n-emb', type=int, default=400)
    parser.add_argument('--n-mdl', type=int, default=1200)
    args = parser.parse_args()
    train(args)


