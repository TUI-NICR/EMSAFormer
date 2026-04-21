#!/bin/bash
set -eo pipefail


prepare_datasets() {
    # NYUv2
    echo "Preparing NYUv2 dataset (~2 GB)"
    nicr_sa_prepare_dataset \
        nyuv2 \
        ./datasets/nyuv2

    # SUNRGB-D
    echo "Preparing SUNRGB-D dataset (~4 GB)"
    nicr_sa_prepare_dataset \
        sunrgbd \
        ./datasets/sunrgbd \
        --create-instances \
        --instances-version emsanet \
        --copy-instances-from-nyuv2 \
        --nyuv2-path ./datasets/nyuv2/

    # ScanNet
    echo "Preparing ScanNet dataset (requires raw ScanNet download)"
    nicr_sa_prepare_dataset \
        scannet \
        /path/where/to/download/ScanNet \
        ./datasets/scannet \
        --n-processes 16 \
        --subsample 50 \
        --additional-subsamples 100 200 500
}


prepare_datasets_nyuv2_internal() {
    # NYUv2
    echo "Preparing NYUv2 dataset (~2 GB)"
    nicr_sa_prepare_dataset \
        nyuv2 \
        ./datasets/nyuv2 \
        --mat-filepath /datasets_nas/segmentation/nyuv2/nyu_depth_v2_labeled.mat \
        --enable-normal-extraction \
        --normal-filepath /datasets_nas/segmentation/nyuv2/normals_gt.tgz
}


prepare_datasets_sunrgbd_internal() {
    # SUNRGB-D
    echo "Preparing SUNRGB-D dataset (~4 GB)"
    nicr_sa_prepare_dataset \
        sunrgbd \
        ./datasets/sunrgbd \
        --create-instances \
        --instances-version emsanet \
        --copy-instances-from-nyuv2 \
        --nyuv2-path ./datasets/nyuv2/ \
        --toolbox-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDtoolbox.zip \
        --data-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBD.zip \
        --box-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDMeta3DBB_v2.mat
}


prepare_datasets_scannet_internal() {
    # ScanNet
    echo "Preparing ScanNet dataset"
    nicr_sa_prepare_dataset \
        scannet \
        /datasets_nas/segmentation/ScanNet/ \
        ./datasets/scannet \
        --n-processes 16 \
        --subsample 50 \
        --additional-subsamples 100 200 500
}


prepare_datasets_internal() {
    rm -rf ./datasets
    mkdir -p /local/emsaformer_datasets
    ln -s /local/emsaformer_datasets ./datasets

    prepare_datasets_nyuv2_internal
    prepare_datasets_sunrgbd_internal
    prepare_datasets_scannet_internal
}


prepare_datasets
# prepare_datasets_internal
