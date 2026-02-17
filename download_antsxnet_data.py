#!/usr/bin/env python

import antspynet
import argparse
import os
import sys
import tensorflow as tf

def download_data(strict=False, cache_dir=None, data_keys=None, model_keys=None, verbose=False):
    """Download data and / or networks.

    If called with no arguments, this will attempt to download all data and networks to the default cache directory.

    Args:
        strict (bool, optional): Exit with error if any download fails
        cache_dir (_str_, optional): Cache directory to use for downloads. If None, the default cache directory
        `~/.keras` will be used.
        data_keys (_list_ of _str_, optional): List of data keys to download. If None, all available data will be downloaded.
        model_keys (_list_ of _str_, optional): List of model keys to download. If None, all available models will be
        downloaded.
        verbose (bool, optional): If True, show more details of downloads. Default is False.
    """
    print("Downloading data files from get_antsxnet_data...")
    if cache_dir is not None:
        antspynet.set_antsxnet_cache_directory(cache_dir)
        print(f"Using custom cache directory: {cache_dir}")
    try:
        if data_keys is None:
            data_keys = antspynet.get_antsxnet_data("show")
        for key in data_keys:
            if key == "show":
                continue
            try:
                print(f"  ↳ Downloading data: {key}")
                fpath = antspynet.get_antsxnet_data(key)
                if verbose:
                    print(f"    ✓ Saved to: {fpath}")
            except Exception as e:
                print(f"    ✗ Failed to download {key}: {e}")
                if strict:
                    raise
    except Exception as e:
        print(f"✗ Failed to retrieve data keys: {e}")
        if strict:
            sys.exit(1)

    print("\nDownloading model weights from get_pretrained_network...")
    try:
        if model_keys is None:
            model_keys = antspynet.get_pretrained_network("show")
        for key in model_keys:
            if key == "show":
                continue
            try:
                print(f"  ↳ Downloading model: {key}")
                fpath = antspynet.get_pretrained_network(key)
                if verbose:
                    print(f"    ✓ Saved to: {fpath}")
            except Exception as e:
                print(f"    ✗ Failed to download {key}: {e}")
                if strict:
                    raise
    except Exception as e:
        print(f"✗ Failed to retrieve model keys: {e}")
        if strict:
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Download ANTsXNet data and / or pretrained models to the local cache directory.

    The default behavior is to download all available data and models to the default cache directory (`~/.keras`).

    Optionally, the cache directory can be customized, and specific data and model keys can be specified.

    If any download fails, the script will continue by default, but this can be changed with the `--strict` flag to exit on the first failure.

    """)
    parser.add_argument("--strict", action="store_true", help="Exit on first failed download.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output showing download progress.")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory for downloads.", default=None)
    parser.add_argument("--data-key-file", type=str, help="Text file containing a list of data keys to download, one per line.",
                        default=None)
    parser.add_argument("--model-key-file", type=str, help="Text file containing a list of model keys to download, one per "
                        "line.", default=None)
    parser.add_argument("--data-keys", nargs='+', type=str, help="One or more data keys to download, separated by spaces.",
                        default=None)
    parser.add_argument("--model-keys", nargs='+',type=str, help="One or more model keys to download, separated by spaces.",
                        default=None)
    args = parser.parse_args()

    if not args.verbose:
        # This stops download progress logs from clogging the output in non-interactive environments
        tf.keras.utils.disable_interactive_logging()

    # Cache dir must be an absolute path, otherwise keras will interpret it relative to ~/.keras/
    if args.cache_dir is not None:
        args.cache_dir = os.path.abspath(args.cache_dir)

    if args.data_key_file is not None and args.data_keys is not None:
        print("Error: Cannot specify both --data-key-file and --data-keys.")
        sys.exit(1)
    if args.model_key_file is not None and args.model_keys is not None:
        print("Error: Cannot specify both --model-key-file and --model-keys.")
        sys.exit(1)

    data_keys = args.data_keys

    if args.data_key_file is not None:
        if not os.path.isfile(args.data_key_file):
            print(f"Error: Data keys file '{args.data_key_file}' does not exist.")
            sys.exit(1)
        with open(args.data_key_file, "r") as f:
            data_keys = [line.strip() for line in f if line.strip()]

    model_keys = args.model_keys

    if args.model_key_file is not None:
        if not os.path.isfile(args.model_key_file):
            print(f"Error: Model keys file '{args.model_key_file}' does not exist.")
            sys.exit(1)
        with open(args.model_key_file, "r") as f:
            model_keys = [line.strip() for line in f if line.strip()]

    try:
        download_data(strict=args.strict, cache_dir=args.cache_dir, data_keys=data_keys, model_keys=model_keys,
                      verbose=args.verbose)
    except Exception as e:
        print(f"\nAborted due to error: {e}")
        sys.exit(1)
