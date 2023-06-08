# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os

from nicr_mt_scene_analysis.testing.onnx import export_onnx_model
import pytest
import torch

from emsaformer.args import ArgParserEMSAFormer
from emsaformer.data import get_dataset
from emsaformer.model import EMSAFormer


def model_test(tasks,
               panoptic_enabled,
               modalities,
               backbone,
               do_postprocessing,
               training,
               tmp_path,
               additional_args=[]):

    parser = ArgParserEMSAFormer()
    args = parser.parse_args([
        '--input-modalities', *modalities,
        '--tasks', *tasks,
        '--instance-encoder-decoder-fusion', 'swin-add-rgb' if len(modalities) > 1 else 'swin-add',
        '--semantic-encoder-decoder-fusion', 'swin-select-rgb' if len(modalities) > 1 else 'swin-select',
        '--normal-encoder-decoder-fusion', 'swin-add-rgb' if len(modalities) > 1 else 'swin-add',
        '--rgbd-encoder-backbone', backbone,
        '--rgb-encoder-backbone', backbone,
        '--depth-encoder-backbone', backbone,
        '--no-pretrained-backbone',
        '--dataset', 'nyuv2',
        *additional_args
    ], verbose=False)

    # replace some args
    args.enable_panoptic = panoptic_enabled

    dataset = get_dataset(args, split='train')
    dataset_config = dataset.config

    # create model
    model = EMSAFormer(args, dataset_config=dataset_config)
    if not training:
        model.eval()

    # determine input
    batch_size = 3
    input_shape = (480, 640)
    batch = {}
    if 'rgb' in args.input_modalities or 'rgbd' in args.input_modalities:
        batch['rgb'] = torch.randn((batch_size, 3)+input_shape)
    if 'depth' in args.input_modalities or 'rgbd' in args.input_modalities:
        batch['depth'] = torch.randn((batch_size, 1)+input_shape)
    if 'instance' in tasks:
        # pure instance segmentation task requires gt foreground mask
        batch['instance_foreground'] = torch.ones(
            (batch_size, 1)+input_shape,
            dtype=torch.bool
        )
    if 'orientation' in tasks:
        # orientation estimation requires a gt segmentation and foreground mask
        batch['instance'] = torch.ones(
            (batch_size, 1)+input_shape,
            dtype=torch.bool
        )
        batch['orientation_foreground'] = torch.ones(
            (batch_size, 1)+input_shape,
            dtype=torch.bool
        )

    if not training and do_postprocessing:
        # for inference postprocessing, inputs in full resolution are required
        if 'rgb' in batch:
            batch['rgb_fullres'] = batch['rgb'].clone()
        if 'depth' in batch:
            batch['depth_fullres'] = batch['depth'].clone()

    # apply model
    outputs = model(batch, do_postprocessing=do_postprocessing)

    # some simple checks for output
    if do_postprocessing:
        assert isinstance(outputs, dict)
    else:
        assert isinstance(outputs, list)
    assert outputs

    # export model to ONNX
    if not training and do_postprocessing:
        # stop here: inference postprocessing is challenging (no onnx export)
        return
    # determine filename and filepath
    tasks_str = '+'.join(tasks)
    if panoptic_enabled:
        tasks_str += '+panoptic'
    modalities_str = '+'.join(modalities)
    filename = f'model_{modalities_str}_{tasks_str}'
    filename += f'__backbone_{backbone}'
    filename += f'__train{training}'
    filename += f'__post_{do_postprocessing}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    # export
    # note, the last element in input tuple is interpreted as named args
    # if no named args should be passed use
    x = (batch, {'do_postprocessing': do_postprocessing})
    export_onnx_model(filepath, model, x)


@pytest.mark.parametrize('tasks', (('semantic',),
                                   ('semantic', 'instance'),
                                   ('semantic', 'instance', 'orientation'),
                                   ('semantic', 'instance', 'orientation',
                                    'scene', 'normal')))
@pytest.mark.parametrize('modalities', (('rgbd',),
                                        ('depth',),
                                        ('rgb', 'depth'),
                                        ('rgbd',)))
@pytest.mark.parametrize('backbone', ('swin-t', 'swin-t-v2',
                                      'swin-t-128', 'swin-t-v2-128'))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_model(tasks, modalities, backbone, do_postprocessing,
               training, tmp_path):
    """Test entire EMSAFormer model"""
    model_test(tasks, False, modalities, backbone,
               do_postprocessing, training, tmp_path)


@pytest.mark.parametrize('tasks', (('semantic', 'instance'),
                                   ('semantic', 'instance', 'orientation'),
                                   ('semantic', 'instance', 'orientation',
                                    'scene', 'normal')))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_model_panoptic(tasks, do_postprocessing, training, tmp_path):
    """Test entire EMSAFormer model (panoptic - single encoder)"""
    model_test(
        tasks=tasks,
        panoptic_enabled=True,
        modalities=('rgbd',),
        backbone='swin-multi-t-v2-128',
        do_postprocessing=do_postprocessing,
        training=training,
        tmp_path=tmp_path
    )


@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_model_less_downsampling_skips(do_postprocessing, training, tmp_path):
    """Test EMSAFormer model with less downsampling and less skip connections"""
    model_test(
        tasks=('semantic', 'instance'),
        panoptic_enabled=True,
        modalities=('rgbd',),
        backbone='swin-multi-t-v2-128',
        do_postprocessing=do_postprocessing,
        training=training,
        tmp_path=tmp_path,
        additional_args=[
            '--semantic-decoder-n-blocks', '1',
            '--instance-decoder-n-blocks', '1',
            '--encoder-decoder-skip-downsamplings', '4', '8',
            '--semantic-decoder-n-channels', '256', '128', '64',
            '--instance-decoder-n-channels', '256', '128', '64'
        ]
    )
