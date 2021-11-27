import torch
import torch.nn as nn

from ...utils import box_coder_utils, box_utils, common_utils
from .point_head_template import PointHeadTemplate
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules


class PointHeadBoxBased(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, voxel_size, point_cloud_range, backbone_channels, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training
        # self.MSGF_cfg = self.model_cfg.MSGF
        # layers_cfg = self.MSGF_cfg.LAYERS

        # c_out = 0
        # self.multi_scale_grid_feature_layers = nn.ModuleList()
        # for src_name in self.MSGF_cfg.FEATURES_SOURCE:
        #     mlps = layers_cfg[src_name].MLPS
        #     for k in range(len(mlps)):
        #         mlps[k] = [backbone_channels[src_name]] + mlps[k]
        #     pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
        #         query_ranges=layers_cfg[src_name].QUERY_RANGES,
        #         nsamples=layers_cfg[src_name].NSAMPLE,
        #         radii=layers_cfg[src_name].POOL_RADIUS,
        #         mlps=mlps,
        #         pool_method=layers_cfg[src_name].POOL_METHOD,
        #     )
        #
        #     self.multi_scale_grid_feature_layers.append(pool_layer)
        #
        #     c_out += sum([x[-1] for x in mlps])

        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=256,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=256,
            output_channels=self.box_coder.code_size
        )

        self.point_feature_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.FEATURES_FC,
            input_channels=16,
            output_channels=128,
            end_bn_active=True,
        )
        self.semantic_geo_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.SEM_GEO_FC,
            input_channels=256,
            output_channels=256,
            end_bn_active=True,
        )
        self.pos_encode_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.POS_FC,
            input_channels=3,
            output_channels=256,
            end_bn_active=True,
        )
        self.encoded_feature_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.EF_FC,
            input_channels=256,
            output_channels=128,
            end_bn_active=True,
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        batch_idxs = keypoints[:, 0]

        point_bev_features_list = []
        for k in range(batch_size):
            batch_mask = batch_idxs == k
            cur_x_idxs = x_idxs[batch_mask]
            cur_y_idxs = y_idxs[batch_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_coords = batch_dict['point_coords']
        batch_size = batch_dict['batch_size']
        # point_indices = batch_dict['point_indices']

        # point_batch_cnt = point_features.new_zeros(batch_size).int()
        # for bs_idx in range(batch_size):
        #     point_batch_cnt[bs_idx] = (point_coords[:, 0] == bs_idx).sum()
        # multi_scale_grid_features_list = []
        # for k, src_name in enumerate(self.MSGF_cfg.FEATURES_SOURCE):
        #     multi_scale_grid_feature_layers = self.multi_scale_grid_feature_layers[k]
        #     cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
        #     cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
        #
        #     # compute voxel center xyz and batch_cnt
        #     cur_coords = cur_sp_tensors.indices
        #     cur_voxel_xyz = common_utils.get_voxel_centers(
        #         cur_coords[:, 1:4],
        #         downsample_times=cur_stride,
        #         voxel_size=self.voxel_size,
        #         point_cloud_range=self.point_cloud_range
        #     )
        #     cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
        #     for bs_idx in range(batch_size):
        #         cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
        #     # get voxel2point tensor
        #     v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
        #     # compute the grid coordinates in this scale, in [b, x, y, z] order
        #     cur_point_grid_coords = point_indices[:, 1:] // cur_stride
        #     cur_point_grid_coords = torch.cat([point_indices[:, 0:1], cur_point_grid_coords], dim=-1)
        #     cur_point_grid_coords = cur_point_grid_coords.int()
        #     # voxel neighbor aggregation
        #     cur_multi_scale_grid_features = multi_scale_grid_feature_layers(
        #         xyz=cur_voxel_xyz.contiguous(),
        #         xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
        #         new_xyz=point_coords[:, 1:].contiguous(),
        #         new_xyz_batch_cnt=point_batch_cnt,
        #         new_coords=cur_point_grid_coords.contiguous().view(-1, 4),
        #         features=cur_sp_tensors.features.contiguous(),
        #         voxel2point_indices=v2p_ind_tensor,
        #         new_coords_bzyx=True
        #     )
        #     multi_scale_grid_features_list.append(cur_multi_scale_grid_features)
        # multi_scale_grid_features = torch.cat(multi_scale_grid_features_list, dim=-1)

        encoded_features = batch_dict['spatial_features_2d']
        encoded_tensor_stride = batch_dict['spatial_features_stride']
        encoded_features_to_pointwise = self.interpolate_from_bev_features(
            keypoints=point_coords,
            bev_features=encoded_features,
            batch_size=batch_size,
            bev_stride=encoded_tensor_stride,
        )
        pos_encode = self.pos_encode_layers(point_coords[:, 1:])
        encoded_features_to_pointwise = self.encoded_feature_layers(encoded_features_to_pointwise + pos_encode)
        point_features = self.point_feature_layers(point_features)

        semantic_geo_features = torch.cat([encoded_features_to_pointwise,
                                           point_features,], dim=1)
        semantic_geo_features = self.semantic_geo_layers(semantic_geo_features)

        point_cls_preds = self.cls_layers(semantic_geo_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(semantic_geo_features)  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)
        batch_dict['semantic_geo_features'] = semantic_geo_features
        # batch_dict['multi_scale_3d_features']['x_up1'].features = semantic_geo_features

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

            # batch_cls_preds_voxel = batch_dict['batch_cls_preds']
            # batch_box_preds_voxel = batch_dict['batch_box_preds']
            # batch_index_voxel = batch_cls_preds_voxel.new_zeros((batch_cls_preds_voxel.shape[0:2]))
            # for i in range(batch_size):
            #     batch_index_voxel[i, :] = i
            # batch_cls_preds_voxel = batch_cls_preds_voxel.view(-1, 3)
            # batch_box_preds_voxel = batch_box_preds_voxel.view(-1, 7)
            # batch_index_voxel = batch_index_voxel.view(-1)
            # batch_dict['batch_cls_preds'] = torch.cat([batch_cls_preds_voxel, point_cls_preds], dim=0)
            # batch_dict['batch_box_preds'] = torch.cat([batch_box_preds_voxel, point_box_preds], dim=0)
            # batch_dict['batch_index'] = torch.cat([batch_index_voxel, batch_dict['point_coords'][:, 0]], dim=0)
            # batch_dict['cls_preds_normalized'] = False



        self.forward_ret_dict = ret_dict

        return batch_dict
