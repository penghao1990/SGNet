import torch
import torch.nn as nn
import numpy as np

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch

# from mayavi import mlab


class PointIntraPartOffsetBasedHead(PointHeadTemplate):
    """
    Point-based head for predicting the intra-object part locations.
    Reference Paper: https://arxiv.org/abs/1907.03670
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, voxel_size, point_cloud_range, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=256,
            output_channels=num_class
        )
        self.part_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.PART_FC,
            input_channels=256,
            output_channels=3
        )
        target_cfg = self.model_cfg.TARGET_CONFIG
        if target_cfg.get('BOX_CODER', None) is not None:
            self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
                **target_cfg.BOX_CODER_CONFIG
            )
            self.box_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.REG_FC,
                input_channels=256,
                output_channels=self.box_coder.code_size
            )
        else:
            self.box_layers = None

        self.point_feature_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.FEATURES_FC,
            input_channels=64,
            output_channels=128,
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
        self.ef_attention_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.EFA_FC,
            input_channels=256,
            output_channels=1,
        )
        self.pf_attention_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.PFA_FC,
            input_channels=256,
            output_channels=1,
        )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.cls_layers[6].bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.box_layers[6].weight, mean=0, std=0.001)
        nn.init.constant_(self.box_layers[6].bias, 0)

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
        # end_point_coords = input_dict['end_point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords,
            gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=True, ret_box_labels=(self.box_layers is not None)
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)
        point_loss = point_loss_cls + point_loss_part

        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
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

    # def show_point_score(self, batch_size, point_coords, cls_scores):
    #     # point_coords = common_utils.get_voxel_centers(
    #     #     sp_tensor.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
    #     #     point_cloud_range=self.point_cloud_range
    #     # )
    #     # point_coords = torch.cat((sp_tensor.indices[:, 0:1].float(), point_coords), dim=1)
    #     # cls_scores = cls_scores[mask]
    #     # point_part_scores = point_part_scores[mask]
    #     for i in range(0, batch_size):
    #         mlab.figure(bgcolor=(0, 0, 0))
    #         mask = point_coords[:, 0] == i
    #         cur_batch_point_coords = point_coords[mask]
    #         cur_x = cur_batch_point_coords[:, 1].cpu().numpy()
    #         cur_y = cur_batch_point_coords[:, 2].cpu().numpy()
    #         cur_z = cur_batch_point_coords[:, 3].cpu().numpy()
    #         # val_depict = (torch.sigmoid(val[mask])).detach().cpu().numpy()
    #         val_depict = (cls_scores[mask]).detach().cpu().numpy()
    #
    #         mlab.points3d(cur_x, cur_y, cur_z, val_depict, scale_factor=0.25, scale_mode='none', mode='sphere',
    #                       line_width=1, colormap='jet')
    #         mlab.colorbar()
    #         mlab.show()

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['point_features']
        point_coords = batch_dict['point_coords']
        batch_size = batch_dict['batch_size']
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
        # encoded_features_to_pointwise = self.encoded_feature_layers(encoded_features_to_pointwise)
        point_features = self.point_feature_layers(point_features)

        semantic_geo_features_base = torch.cat([encoded_features_to_pointwise,
                                                point_features, ], dim=1)
        encoded_features_to_pointwise_attention = torch.sigmoid(self.ef_attention_layers(semantic_geo_features_base))
        point_features_attention = torch.sigmoid(self.pf_attention_layers(semantic_geo_features_base))
        encoded_features_to_pointwise = encoded_features_to_pointwise*encoded_features_to_pointwise_attention
        point_features = point_features*point_features_attention
        semantic_geo_features = torch.cat([encoded_features_to_pointwise,
                                                point_features, ], dim=1)
        # semantic_geo_features = self.semantic_geo_layers(semantic_geo_features)

        point_cls_preds = self.cls_layers(semantic_geo_features)  # (total_points, num_class)
        point_part_preds = self.part_reg_layers(semantic_geo_features)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            'point_part_preds': point_part_preds,
        }
        if self.box_layers is not None:
            point_box_preds = self.box_layers(semantic_geo_features)
            ret_dict['point_box_preds'] = point_box_preds

        # _, point_coords_wlhr = self.generate_predicted_boxes(
        #     points=batch_dict['point_coords'][:, 1:4],
        #     point_cls_preds=point_cls_preds, point_box_preds=ret_dict['point_box_preds']
        # )

        # end_point_coords = torch.cat([point_coords[:, 0:1], point_coords_wlhr[:, 0:3]],
        #                              dim=1).detach()
        # batch_dict['end_point_coords'] = end_point_coords

        point_cls_scores = torch.sigmoid(point_cls_preds)
        point_part_offset = torch.sigmoid(point_part_preds)
        # score_for_show = point_part_offset.mean(dim=1)*point_cls_scores.max(dim=-1)[0]
        # self.show_point_score(
        #     batch_size=2,
        #     point_coords=point_coords,
        #     cls_scores=score_for_show
        # )
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        batch_dict['point_part_offset'] = point_part_offset
        batch_dict['multi_scale_3d_features']['x_points'].features = semantic_geo_features

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_part_labels'] = targets_dict.get('point_part_labels')
            ret_dict['point_box_labels'] = targets_dict.get('point_box_labels')

        if self.box_layers is not None and (not self.training or self.predict_boxes_when_training):
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=ret_dict['point_box_preds']
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict
        return batch_dict
