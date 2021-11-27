from functools import partial
import torch
import spconv
import torch.nn as nn
from ...utils import common_utils


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        # self.convz = spconv.SparseSequential(
        #     # block(16, 32, (3, 1, 1), norm_fn=norm_fn, padding=(1, 0, 0), indice_key='subm1', conv_type='spconv'),
        #     block(16, 16, (1, 3, 3), norm_fn=norm_fn, padding=(0, 1, 1), indice_key='submxy', conv_type='spconv'),
        #     # block(16, 16, (1, 3, 3), norm_fn=norm_fn, padding=(0, 1, 1), indice_key='submxy'),
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='spconv'),
        # )
        # self.convx = spconv.SparseSequential(
        #     # block(16, 32, (1, 1, 3), norm_fn=norm_fn, padding=(0, 0, 1), indice_key='submx', conv_type='spconv'),
        #     block(16, 16, (3, 3, 1), norm_fn=norm_fn, padding=(1, 1, 0), indice_key='submzy',conv_type='spconv'),
        #     # block(16, 16, (3, 3, 1), norm_fn=norm_fn, padding=(1, 1, 0), indice_key='submzy'),
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='spconv'),
        # )
        # self.convy = spconv.SparseSequential(
        #     # block(16, 32, (1, 3, 1), norm_fn=norm_fn, padding=(0, 1, 0), indice_key='submy', conv_type='spconv'),
        #     block(16, 16, (3, 1, 3), norm_fn=norm_fn, padding=(1, 0, 1), indice_key='submzx',conv_type='spconv'),
        #     # block(16, 16, (3, 1, 3), norm_fn=norm_fn, padding=(1, 0, 1), indice_key='submzx'),
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='spconv'),
        # )

        self.conv_points = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )

        # self.points_out = spconv.SparseSequential(
        #     # block(48, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='spconv'),
        #     block(48, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        # )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 64
        self.backbone_channels = {
            # 'x_conv1': 256,
            # 'x_conv2': 32,
            # 'x_conv3': 64,
            # 'x_conv4': 64,
            'x_points': 256
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # x_convz = self.convz(x_conv1)
        # x_convx = self.convx(x_conv1)
        # x_convy = self.convy(x_convx)
        # x_convz = self.convz(x_convy)
        # x_convz.features = torch.cat([x_convx.features, x_convy.features, x_convz.features], dim=-1)

        x_points = self.conv_points(x_conv1)
        # x_convx = self.convx(x_conv1)
        # x_convy = self.convy(x_conv1)
        # x_convz = self.convz(x_conv1)
        # x_convpoint= self.conv_points(x_conv1)
        # x_convpoint.features = torch.cat([x_convx.features, x_convy.features, x_convz.features, x_convpoint.features], dim=-1)
        # x_points_out = self.points_out(x_convz)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                # 'x_conv1': x_conv1,
                # 'x_conv2': x_conv2,
                # 'x_conv3': x_conv3,
                # 'x_conv4': x_conv4,
                'x_points': x_points,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                # 'x_conv1': 1,
                # 'x_conv2': 2,
                # 'x_conv3': 4,
                # 'x_conv4': 8,
                'x_points': 1,
            }
        })

        batch_dict['point_features'] = x_points.features
        point_coords = common_utils.get_voxel_centers(
            x_points.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x_points.indices[:, 0:1].float(), point_coords), dim=1)
        batch_dict['point_indices'] = x_points.indices

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)



        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict['point_features'] = x_conv1.features
        point_coords = common_utils.get_voxel_centers(
            x_conv1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x_conv1.indices[:, 0:1].float(), point_coords), dim=1)
        batch_dict['point_indices'] = x_conv1.indices

        return batch_dict
