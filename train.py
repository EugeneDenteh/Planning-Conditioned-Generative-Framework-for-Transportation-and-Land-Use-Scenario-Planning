import gc
import torch
from src.engine import arguments, run
from src.distributed import finish


def sequential(args):
    args.stage = 1
    first = run(args)
    args.stage = 2
    args.resume = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    second = run(args)
    return first, second


if __name__ == "__main__":
    try:
        sequential(arguments().parse_args())
    finally:
        finish()
