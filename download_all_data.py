import antspynet
import argparse
import sys
import tensorflow as tf

def download_all_data(strict=False, cache_dir=None):
    print("Downloading data files from get_antsxnet_data...")
    if cache_dir is not None:
        antspynet.set_antsxnet_cache_directory(cache_dir)
        print(f"Using custom cache directory: {cache_dir}")
    try:
        data_keys = antspynet.get_antsxnet_data("show")
        for key in data_keys:
            if key == "show":
                continue
            try:
                print(f"  ↳ Downloading data: {key}")
                fpath = antspynet.get_antsxnet_data(key)
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
        model_keys = antspynet.get_pretrained_network("show")
        for key in model_keys:
            if key == "show":
                continue
            try:
                print(f"  ↳ Downloading model: {key}")
                fpath = antspynet.get_pretrained_network(key)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit on first failed download.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output showing download progress.")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory for downloads.", default=None)
    args = parser.parse_args()

    if not args.verbose:
        # This stops download progress logs from clogging the output in non-interactive environments
        tf.keras.utils.disable_interactive_logging()

    try:
        download_all_data(strict=args.strict, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"\nAborted due to error: {e}")
        sys.exit(1)
