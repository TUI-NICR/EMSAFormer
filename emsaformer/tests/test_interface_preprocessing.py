# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from functools import partial

from nicr_mt_scene_analysis.data import mt_collate
from nicr_mt_scene_analysis.data import CollateIgnoredDict
from nicr_mt_scene_analysis.testing.preprocessing import show_results
from nicr_mt_scene_analysis.testing.preprocessing import SHOW_RESULTS
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT
import numpy as np
import pytest
import torch

from emsaformer.args import ArgParserEMSAFormer
from emsaformer.data import get_dataset
from emsaformer.preprocessing import get_preprocessor


@pytest.mark.parametrize('dataset', ('nyuv2', 'sunrgbd', 'hypersim', 'scannet'))
@pytest.mark.parametrize('tasks', (('semantic',),
                                   ('semantic', 'instance'),
                                   ('instance', 'orientation'),
                                   ('semantic', 'instance', 'orientation'),
                                   ('semantic', 'instance', 'orientation',
                                    'scene', 'normal')))
@pytest.mark.parametrize('modalities', (('rgb',),
                                        ('depth',),
                                        ('rgb', 'depth')))
@pytest.mark.parametrize('phase', (('train', 'test')))
@pytest.mark.parametrize('multiscale', (False, True))
def test_preprocessing(dataset, tasks, modalities, phase, multiscale):
    """Test entire EMSANet/EMSAFormer preprocessing"""

    # drop normal task for SUNRGB-D
    if dataset not in ('hypersim', 'nyuv2'):
        tasks = tuple(t for t in tasks if t != 'normal')

    parser = ArgParserEMSAFormer()
    additional_args = []
    if multiscale:
        # ensure emsanet is used for all decoders as the (default) segformer
        # mlp decoder does not support multiscale
        additional_args.extend([
            '--semantic-decoder', 'emsanet',
            '--instance-decoder', 'emsanet',
            '--normal-decoder', 'emsanet'
        ])
    args = parser.parse_args(additional_args, verbose=False)

    args.tasks = tasks
    args.input_modalities = modalities
    args.dataset = dataset
    args.dataset_path = DATASET_PATH_DICT[dataset]
    if dataset in ('cityscapes', 'hypersim', 'scannet'):
        args.raw_depth = True
    if dataset == 'scannet':
        # to test SemanticClassMapper
        args.validation_scannet_benchmark_mode = True

    dataset = get_dataset(args, 'train')

    preprocessor = get_preprocessor(
        args=args,
        dataset=dataset,
        phase=phase,
        multiscale_downscales=(8, 16, 32) if multiscale else None
    )
    dataset.preprocessor = preprocessor

    for sample_pre in dataset:
        if SHOW_RESULTS:
            # use 'SHOW_RESULTS=true pytest ...'
            sample = sample_pre.pop('_no_preprocessing')
            show_results(sample, sample_pre, "Preprocessing")
        else:
            break

    show_results({}, sample_pre, "Preprocessing")

    # we use a modified collate function to handle elements of different
    # spatial resolution and to ignore numpy arrays, dicts containing
    # orientations (OrientationDict), and simple tuples storing shapes
    collate_fn = partial(mt_collate,
                         type_blacklist=(np.ndarray,
                                         CollateIgnoredDict,
                                         OrientationDict,
                                         SampleIdentifier))

    # test with data loader (and collate function)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    for sample_pre in loader:
        break
