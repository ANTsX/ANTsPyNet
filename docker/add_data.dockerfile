FROM antsx/antspynet:latest

USER root
WORKDIR /home/antspyuser

RUN . "${VIRTUAL_ENV}/bin/activate" && \
    /opt/bin/download_antsxnet_data.py \
        --cache-dir /home/antspyuser/.keras/ANTsXNet \
        --strict && \
    chmod -R 0755 /home/antspyuser/.antspy /home/antspyuser/.keras

USER antspyuser

LABEL maintainer="Philip A Cook (https://github.com/cookpa)" \
      description="ANTsPyNet is part of the ANTsX ecosystem (https://github.com/ANTsX). \
ANTsX citation: https://pubmed.ncbi.nlm.nih.gov/33907199"
ENTRYPOINT ["python"]
