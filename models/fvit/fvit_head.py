import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.roi_heads import StandardRoIHead, ConvFCBBoxHead
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.core import bbox2roi, bbox2result
from mmcv.runner import force_fp32
from mmdet.core import multiclass_nms


@HEADS.register_module()
class FViTBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 fixed_temperature=0,
                 learned_temperature=50.0,
                 class_embed=None,
                 seen_classes=None,
                 all_classes=None,
                 vlm_temperature=100.0,
                 alpha=0.2,
                 beta=0.45,
                 learn_bg=False,
                 **kwargs):
        super(FViTBBoxHead, self).__init__(**kwargs)
        if fixed_temperature != 0:
            self.detect_temperature = fixed_temperature
        else:
            self.detect_temperature = torch.nn.Parameter(
                torch.tensor(learned_temperature))
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta
        self.learn_bg = learn_bg

        # ----------------- load class embed -----------------
        assert class_embed is not None, 'class embed is None'
        seen_classes = json.load(open(seen_classes)) + ['background']
        all_classes = json.load(open(all_classes)) + ['background']
        idx = [all_classes.index(seen) for seen in seen_classes]

        self.base_idx = torch.zeros(len(all_classes), dtype=bool)
        self.base_idx[idx] = True
        self.novel_idx = self.base_idx == False

        class_embed = torch.load(class_embed)
        all_embed = [class_embed[name] for name in all_classes]
        all_embed = torch.stack(all_embed, dim=0).permute(1, 0).contiguous()
        all_embed = F.normalize(all_embed, p=2, dim=0)
        self.register_buffer('all_embeddings', all_embed)
        if learn_bg:
            self.bg_embedding = nn.Parameter(all_embed[:, -1:])

    @property
    def all_embed(self):
        if self.learn_bg:
            bg_embed = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg_embed], dim=1)
        else:
            return self.all_embeddings

    def forward(self, x, vlm_box_feats=None):
        all_embed = self.all_embed.type_as(x)

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
            x_cls = self.relu(fc(x_cls))  # FIXME: should discard the last relu

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        normalized_x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
        cls_score = normalized_x_cls @ all_embed * self.detect_temperature

        if not self.training:
            cls_score = cls_score.softmax(dim=-1)
            vlm_score = vlm_box_feats @ all_embed * self.vlm_temperature
            vlm_score = vlm_score.softmax(dim=-1)

            cls_score[:, self.base_idx] = cls_score[:, self.base_idx] ** (
                    1 - self.alpha) * vlm_score[:, self.base_idx] ** self.alpha
            cls_score[:, self.novel_idx] = cls_score[:, self.novel_idx] ** (
                    1 - self.beta) * vlm_score[:, self.novel_idx] ** self.beta

        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            if self.training:
                scores = F.softmax(cls_score,
                                   dim=-1) if cls_score is not None else None
            else:
                scores = cls_score
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
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

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels


@HEADS.register_module()
class FViTRoIHead(StandardRoIHead):
    def __init__(self,
                 vlm_roi_extractor=None,
                 add_novel_bbox=False,
                 select_topk_novel=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.vlm_roi_extractor = build_roi_extractor(vlm_roi_extractor)
        self.add_novel_bbox = add_novel_bbox
        self.select_topk_novel = select_topk_novel

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    vlm_feat=None,
                    proposals=None,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, vlm_feat=vlm_feat)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           vlm_feat=None,
                           rescale=False):
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois, vlm_feat=vlm_feat)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def _bbox_forward(self, x, rois, vlm_feat=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        vlm_roi_feats = None
        if vlm_feat is not None:
            vlm_roi_feats = self.vlm_roi_extractor([vlm_feat], rois)[..., 0, 0]
            vlm_roi_feats = F.normalize(vlm_roi_feats, dim=-1, p=2)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, vlm_roi_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, vlm_roi_feats=vlm_roi_feats)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      vlm_feat=None,
                      **kwargs):
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        if not self.add_novel_bbox:
            vlm_feat = None
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x,
                sampling_results,
                gt_bboxes,
                gt_labels,
                img_metas,
                vlm_feat=vlm_feat)
            losses.update(bbox_results['loss_bbox'])
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(
                x, sampling_results, bbox_results['bbox_feats'], gt_masks,
                img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    
    def _bbox_forward_train(self,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            img_metas,
                            vlm_feat=None):
        """Run forward function and calculate loss for box head in training."""

        num_imgs = len(sampling_results)
        
        rois = bbox2roi([res.bboxes for res in sampling_results])
        
        bbox_results = self._bbox_forward(x, rois, vlm_feat=vlm_feat)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, concat=False)
        labels, label_weights, bbox_targets, bbox_weights = bbox_targets

        if self.add_novel_bbox:
            vlm_roi_feats = bbox_results['vlm_roi_feats']
            all_embed = self.bbox_head.all_embed.type_as(vlm_roi_feats)

            if len(all_embed.shape) > 2: # (D, 17, 66)
                shape = all_embed.shape
                vlm_score = vlm_roi_feats @ all_embed.reshape(shape[0], -1) * self.bbox_head.vlm_temperature
                vlm_score = vlm_score.reshape(-1, *shape[1:]).max(1)[0]
            else:
                vlm_score = vlm_roi_feats @ all_embed * self.bbox_head.vlm_temperature
            vlm_score = vlm_score.softmax(-1)
            num_per_img = [len(s.bboxes) for s in sampling_results]
            vlm_score = torch.split(vlm_score, num_per_img)
            for i in range(num_imgs):
                pseudo_label = vlm_score[i].argmax(1)
                is_novel = (vlm_score[i][:, self.bbox_head.novel_idx] == vlm_score[i].max(1)[0][:, None]).any(1)
                is_bg = pseudo_label == self.bbox_head.num_classes
                is_novel[is_bg] = False
                is_novel[:len(sampling_results[i].pos_bboxes)] = False
                is_novel = is_novel & (vlm_score[i].max(-1)[0] > 0.8)
                select_inds = is_novel.nonzero().view(-1)
                if self.select_topk_novel != 0:
                    select_inds = select_inds[:self.select_topk_novel]
                # if self.select_random_k_novel != 0:
                #     select_inds = select_inds[torch.randperm(len(select_inds))[:self.select_random_k_novel]]

                labels[i][select_inds] = pseudo_label[select_inds]
                label_weights[i][select_inds] = 0.2
                
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois, labels, label_weights, bbox_targets, bbox_weights
        )
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results