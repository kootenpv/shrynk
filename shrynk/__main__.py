import pandas as pd
from shrynk import save, load, show_benchmark, run_benchmarks, infer
from shrynk.utils import add_z_to_bench


def main():
    """ This is the function that is run from commandline with `yagmail` """
    import argparse

    parser = argparse.ArgumentParser(
        description='Use the machine learning meta library shrynk to compress'
    )
    subparsers = parser.add_subparsers(dest="command")
    compress = subparsers.add_parser('compress')
    compress.add_argument('file', help='file you want to compress')
    compress.add_argument('--size', '-s', default=3, type=int, help='Size weight for model')
    compress.add_argument('--write', '-w', default=1, type=int, help='Write-time weight for model')
    compress.add_argument('--read', '-r', default=1, type=int, help='Read-time weight for model')
    decompress = subparsers.add_parser('decompress')
    decompress.add_argument('file', help='file you want to decompress')
    benchmark = subparsers.add_parser('benchmark')
    benchmark.add_argument('file', help='file you want to benchmark')
    benchmark.add_argument('--size', '-s', default=3, type=int, help='Size weight for model')
    benchmark.add_argument('--write', '-w', default=1, type=int, help='Write-time weight for model')
    benchmark.add_argument('--read', '-r', default=1, type=int, help='Read-time weight for model')
    benchmark.add_argument('--predict', help='Read-time weight for model', action="store_true")
    benchmark.add_argument('--save', help='Read-time weight for model', action="store_true")
    parser.add_argument(
        '-password', '-p', help='Preferable to use keyring rather than password here'
    )
    args = parser.parse_args()
    if args.command == "compress":
        data = load(args.file)
        save(data, args.file, size=args.size, write=args.write, read=args.read)
    if args.command == "decompress":
        data = load(args.file)
        if "json" in args.file:
            ext = "json"
            kwargs = {"compression": None}
            end = args.file.index("." + ext)
            destination = args.file[:end] + "." + ext
        elif "csv" in args.file or "parquet" in args.file:
            ext = "csv"
            kwargs = {"engine": "csv", "compression": None}
            end = args.file.index("." + ext)
            destination = args.file[:end] + "." + ext
        else:
            kwargs = {"compression": None}
            destination = ".".join(args.file.split(".")[:-1])
        save(data, destination, kwargs)
    elif args.command == "benchmark":
        if args.predict:
            data = load(args.file)
            print("Predicted:", infer(data, size=args.size, write=args.write, read=args.read))
        if args.save:
            bench = run_benchmarks(args.file)
            bench = pd.DataFrame(bench, columns=["kwargs", "size", "write_time", "read_time"])
            return print(add_z_to_bench(bench, args.size, args.write, args.read))
        else:
            print(show_benchmark(args.file, size=args.size, write=args.write, read=args.read))
