import antspynet
import argparse
import sys

def download_all_data(strict=False):
    print("Downloading data files from get_antsxnet_data...")
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
    args = parser.parse_args()

    try:
        download_all_data(strict=args.strict)
    except Exception as e:
        print(f"\nAborted due to error: {e}")
        sys.exit(1)
