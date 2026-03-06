# ANTsPyNet docker

Container for ANTsPyNet with an option to pre-install data and pre-trained networks.

## Pre-built images on Docker Hub

Images are available on the [ANTsPyNet Docker Hub page](https://hub.docker.com/repository/docker/antsx/antspynet/general). The tags ending with "-with-data" include all pre-trained networks and data required to run the utilities.


## Building the container

From `ANTsPyNet/`,

```
docker build \
  -t your_dockerhub_username/antspynet:latest \
  -f docker/Dockerfile \
  .
```

To include all available data in the container image, build with the data included:

```
docker build \
    -t your_dockerhub_username/antspynet:latest \
    -f docker/Dockerfile \
    --build-arg INSTALL_ANTSXNET_DATA=1 \
    .
```


## Running the container

The docker user is `antspyuser`, and the home directory `/home/antspyuser` exists in the
container. The container always has the ANTsPy data, so that you can call `ants.get_data`
and run ANTsPy tests.

Containers built with data will have all ANTsPyNet data and pretrained networks under
`/home/antspyuser/.keras`.


### Apptainer / Singularity usage

Apptainer always runs as the system user, so you will need
```
apptainer run --containall --no-home --home /home/antspyuser antspynet_latest.sif
```
in order for ANTsPy and ANTsPyNet to find built-in data.


## Run time data and pretrained networks

If the data is not built into the container, ANTsPyNet downloads data and networks at run time. 

By default, data is downloaded on demand and stored in a cache directory at
`${HOME}/.keras`. With the default user, attempts to download data at run time will fail
because the directory `/home/antspyuser` is not writeable. This is by design, to prevent
users unknowingly downloading large amounts of data by running a container repeatedly, or
by running many containers in parallel.

Building the container with data provides better containerization. If image size is an issue,
you can write a Docker file building from the `antspynet` base image and include a subset of data /
networks, see the help for the `download_antsxnet_data.py` script.


### Downloading data to a local cache

This is an advanced option for users who need to minimize container size. Beware that
running different ANTsPyNet versions with the same cache directory can lead to conflicts.
If you do this, you should name the cache directory to include the version of ANTsPyNet
you are using, and preferably make it read-only after populating it. To download all data
and networks, run
```
docker run --rm -it antspynet:latest \
   /opt/bin/download_antsxnet_data.py \
     --cache-dir /path/to/local/cache/dir
```

You can also download a subset of data / networks by providing a list of names in a text
file, one per line
```
docker run --rm -it antspynet:latest \
   /opt/bin/download_antsxnet_data.py \
     --cache-dir /path/to/local/cache/dir \
     --data-key-file datakeys.txt \
     --model-key-file modelkeys.txt
```

If the local cache directory is not mounted as `/home/antspyuser/.keras`, runtime scripts
must call
```python
import antspynet
antspynet.set_antsxnet_cache_directory('/path/to/local/cache/dir')
```

### Ensuring cache validity

Running multiple versions of ANTsPyNet with the same cache directory can lead to
conflicts. ANTsPyNet fetches data based on unique identifiers encoded in the source, so it
will never fetch the wrong data even if newer data exists. However, the cached file names
on disk are shared across versions, so for example if
`~/.keras/ANTsXNet/brainExtractionRobustT1.h5` has been downloaded already, it will not be
updated even if ANTsPyNet is updated. Similarly, if a a new version of that file is
downloaded, it will be used by by older ANTsPyNet installations that use the same cache
directory.


