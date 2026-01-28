#!/usr/bin/env python

#
# This gets ANTsXNet data and pretrained networks
#
# Getting all the data ahead of time is optional, by default it is downloaded
# on demand and stored in ~/.keras/ANTsXNet . But for complete reproducibility,
# or for applications lacking Internet access, the data can be downloaded with
# this script.
#
# To use the data in a container, mount the data directory containing keras.json
# as $USER/.keras.
#
#
import sys

if (len(sys.argv) == 1):
    usage = '''
  Usage: {} /path/to/ANTsXNetData [doInstall=1] [dataList.txt] [networkList.txt]

  Second argument can be passed to skip installation in docker files.

  Subsequent arguments, if specified, read a list of things to fetch from a text file.
  This can be used to get a subset of the data / networks.

  Downloads ANTsXNet data and networks to the specified directory.

  The path MUST be absolute or it will be interpreted relative to
  the default ~/.keras
'''
    print(usage.format(sys.argv[0]))

    sys.exit(1)

import antspynet

# Base output dir, make ANTsXNet/ and keras.json under here
output_dir=sys.argv[1]

do_install=1

if len(sys.argv) > 2:
    do_install = int(sys.argv[2])

if do_install == 0:
    # Exit 0, so docker won't think there's an error
    sys.exit(0)

data_path = f"{output_dir}/ANTsXNet"

all_data = list()

if len(sys.argv) > 3:
    with open(sys.argv[3]) as f:
        all_data = f.read().splitlines()
else:
        all_data = list(antspynet.get_antsxnet_data('show'))
        all_data.remove('show')

antspynet.set_antsxnet_cache_directory(data_path)

for entry in all_data:
    print(f"Downloading {entry}")
    try:
        antspynet.get_antsxnet_data(entry)
    except NotImplementedError as e:
        print(f"Failed to download {entry}")

all_networks = list()

if len(sys.argv) > 4:
    with open(sys.argv[4]) as f:
        all_networks = f.read().splitlines()
else:
    all_networks = list(antspynet.get_pretrained_network('show'))
    all_networks.remove('allen_brain_leftright_coronal_mask_weights')
    all_networks.remove('allen_cerebellum_coronal_mask_weights')
    all_networks.remove('allen_cerebellum_sagittal_mask_weights')
    all_networks.remove('allen_sr_weights')
    all_networks.remove('show')

for entry in all_networks:
    print(f"Downloading {entry}")
    try:
        antspynet.get_pretrained_network(entry)
    except NotImplementedError as e:
        print(f"Failed to download {entry}")

