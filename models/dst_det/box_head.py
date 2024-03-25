import json
import torch

from mmdet.core import  multiclass_nms
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
import torch.nn.functional as F

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, bbox_overlaps

import json
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS, build_head, build_loss
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.utils import build_linear_layer

@HEADS.register_module(force=True)
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        # assert (num_shared_convs + num_shared_fcs + num_cls_convs +
        #         num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        # if num_cls_convs > 0 or num_reg_convs > 0:
        #     assert num_shared_fcs == 0
        # if not self.with_cls:
        #     assert num_cls_convs == 0 and num_cls_fcs == 0
        # if not self.with_reg:
        #     assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

@HEADS.register_module()
class FvlmBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 attnpool=None,
                 fixed_temperature=0,
                 learned_temperature=50.0,
                 class_embed=None,
                 seen_classes=None,
                 unseen_classes=None,
                 all_classes=None,
                 vlm_temperature=100.0,
                 alpha=0.2,
                 beta=0.45,
                 with_bbox_score=False,
                 bbox_score_type='BoxIoU',
                 loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
                 **kwargs):
        super(FvlmBBoxHead, self).__init__(**kwargs)
        self.with_bbox_score = with_bbox_score
        self.bbox_score_type = bbox_score_type
        if with_bbox_score:
            self.fc_bbox_score = torch.nn.Sequential(
                torch.nn.Linear(self.cls_last_dim, 1),
            )
            self.loss_bbox_score = build_loss(loss_bbox_score)

        if attnpool is not None:
            self.attnpool = build_head(attnpool)
        else:
            self.attnpool = None
        # AttentionPool2d(224//32, 2048, num_heads=32, output_dim=1024)
        if fixed_temperature != 0:
            self.detect_temperature = fixed_temperature
        else:
            self.detect_temperature = torch.nn.Parameter(
                torch.tensor(learned_temperature))
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta

        assert class_embed is not None, 'class embed is None'
        seen_classes = json.load(open(seen_classes))
        all_classes = json.load(open(all_classes))
        idx = [all_classes.index(seen) for seen in seen_classes]

        base_idx = torch.zeros(len(all_classes), dtype=bool)
        base_idx[idx] = True
        self.base_idx = torch.cat([base_idx, base_idx.new_ones(1)])
        self.novel_idx = self.base_idx == False

        class_embed = torch.load(class_embed)
        all_embed = [class_embed[name] for name in all_classes]
        if 'background' in class_embed:
            void_embed = class_embed['background']
        else:
            # void_embed = torch.nn.Embedding(1, self.fc_out_channels)
            # void_embed = void_embed.weight.repeat((all_embed[0].shape[0], 1))
            void_embed = torch.zeros(1, self.fc_out_channels).repeat((all_embed[0].shape[0], 1))
        all_embed.append(void_embed)

        self.all_embed = torch.stack(all_embed, dim=0).transpose(-1, 0).contiguous()

        if kwargs['loss_cls']['type'] == 'CustomCrossEntropyLoss':
            kwargs['loss_cls'].update(dict(class_weight=self.base_idx.float()))
        self.loss_cls = build_loss(kwargs['loss_cls'])

    def forward(self, x, vlm_box_feats=None):

        all_embed = self.all_embed.type_as(x)

        if vlm_box_feats is not None:
            attnpool_feats = self.attnpool(vlm_box_feats)
            attnpool_feats = F.normalize(attnpool_feats, dim=-1, p=2)
        else:
            attnpool_feats = None

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        x_bbox_score = x
        bbox_score = self.fc_bbox_score(
            x_bbox_score) if self.with_bbox_score else None

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # if self.with_cls:
        #     return cls_score, bbox_pred

        normalized_x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
        shape = all_embed.shape
        if self.training:
            cls_score = normalized_x_cls @ all_embed.view(shape[0], -1) * self.detect_temperature
            if len(shape) > 2: # (2048, 17, 66)
                cls_score = cls_score.view(-1, *shape[1:])
                cls_score = cls_score.max(1)[0]
        else:
            cls_score = normalized_x_cls @ all_embed.view(shape[0], -1) * self.detect_temperature
            vlm_score = attnpool_feats @ all_embed.view(shape[0], -1) * self.vlm_temperature

            if len(shape) > 2: # (2048, 17, 66)
                cls_score = cls_score.view(-1, *shape[1:])
                vlm_score = vlm_score.view(-1, *shape[1:])
                cls_score = cls_score.max(1)[0]
                vlm_score = vlm_score.max(1)[0]

            cls_score = cls_score.softmax(dim=-1)
            vlm_score = vlm_score.softmax(dim=-1)

            cls_score[:, self.base_idx] = cls_score[:, self.base_idx]**(
                1 - self.alpha) * vlm_score[:, self.base_idx]**self.alpha
            cls_score[:, self.novel_idx] = cls_score[:, self.novel_idx]**(
                1 - self.beta) * vlm_score[:, self.novel_idx]**self.beta
        return cls_score, bbox_pred, attnpool_feats, bbox_score

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   bbox_score,
                   rpn_score,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # cls_score is not used.
        # scores = F.softmax(
        #     cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[..., 1:],
                                            bbox_pred,
                                            max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if self.with_bbox_score:
            scores = torch.sqrt(rpn_score * bbox_score.sigmoid())
            scores = torch.cat([scores, torch.zeros_like(scores)], dim=-1)
        else:
            scores = cls_score

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        (labels, label_weights, bbox_targets, bbox_weights, bbox_score_targets,
         bbox_score_weights) = multi_apply(self._get_target_single,
                                           pos_bboxes_list,
                                           neg_bboxes_list,
                                           pos_gt_bboxes_list,
                                           pos_gt_labels_list,
                                           cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_score_targets = torch.cat(bbox_score_targets, 0)
            bbox_score_weights = torch.cat(bbox_score_weights, 0)

        return (labels, label_weights, bbox_targets, bbox_weights,
                bbox_score_targets, bbox_score_weights)

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1

        # Bbox-IoU as target
        bbox_score_targets = pos_bboxes.new_zeros(num_samples)
        bbox_score_weights = pos_bboxes.new_zeros(num_samples)
        if self.with_bbox_score and num_pos > 0:
            if self.bbox_score_type == 'BoxIoU':
                pos_bbox_score_targets = bbox_overlaps(pos_bboxes,
                                                        pos_gt_bboxes,
                                                        is_aligned=True)
            elif self.bbox_score_type == 'Centerness':
                tblr_bbox_coder = build_bbox_coder(
                    dict(type='TBLRBBoxCoder', normalizer=1.0))
                pos_center_bbox_targets = tblr_bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
                valid_targets = torch.min(pos_center_bbox_targets,
                                            -1)[0] > 0
                pos_center_bbox_targets[valid_targets == False, :] = 0
                top_bottom = pos_center_bbox_targets[:, 0:2]
                left_right = pos_center_bbox_targets[:, 2:4]
                pos_bbox_score_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] /
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] /
                        (torch.max(left_right, -1)[0] + 1e-12)))
            bbox_score_targets[:num_pos] = pos_bbox_score_targets
            bbox_score_weights[:num_pos] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, bbox_score_targets, bbox_score_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(
        self,
        cls_score,
        bbox_pred,
        bbox_score,
        rois,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        bbox_score_targets=None,
        bbox_score_weights=None,
        reduction_override=None,
    ):
        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if self.with_bbox_score:
            if bbox_score.numel() > 0:
                losses['loss_bbox_score'] = self.loss_bbox_score(
                    bbox_score.squeeze(-1).sigmoid(),
                    bbox_score_targets,
                    bbox_score_weights,
                    avg_factor=bbox_score_targets.size(0),
                    reduction_override=reduction_override)
        return losses