from src.engine import arguments, run
from src.distributed import finish

if __name__ == "__main__":
    try:
        run(arguments(1).parse_args())
    finally:
        finish()
