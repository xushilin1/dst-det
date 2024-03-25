import json
import torch

from mmcv.cnn.bricks.transformer import build_transformer_layer, BaseTransformerLayer
from mmdet.models.roi_heads import StandardRoIHead, Shared2FCBBoxHead, ConvFCBBoxHead
from mmdet.models.builder import HEADS, build_roi_extractor, build_head, build_loss
from mmdet.core import (bbox2roi, bbox2result, roi2bbox, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from models.dst_det.resnet import AttentionPool2d
from mmdet.models.losses import accuracy
import torch.nn.functional as F

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, bbox_overlaps





@HEADS.register_module()
class FvlmRoIHead(StandardRoIHead):
    def __init__(self,
                 vlm_roi_extractor=None,
                 backbone_name='RN50',
                 loss_rpn_plus=None,
                 remove_novel_bbox=False,
                 add_novel_bbox=False,
                 neg_only=True,
                 select_topk_novel=0,
                 select_random_k_novel=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.vlm_roi_extractor = build_roi_extractor(vlm_roi_extractor)
        self.loss_rpn_plus = build_loss(
            loss_rpn_plus) if loss_rpn_plus is not None else None
        self.remove_novel_bbox = remove_novel_bbox
        self.add_novel_bbox = add_novel_bbox
        self.neg_only = neg_only
        self.select_topk_novel=select_topk_novel
        self.select_random_k_novel = select_random_k_novel

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    res_feats=None,
                    proposals=None,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(x,
                                                         img_metas,
                                                         proposal_list,
                                                         self.test_cfg,
                                                         res_feats=res_feats,
                                                         rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(x,
                                                 img_metas,
                                                 det_bboxes,
                                                 det_labels,
                                                 rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           res_feats=None,
                           rescale=False):
        
        rpn_score = torch.cat([p[:, -1:] for p in proposals], 0)
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

        bbox_results = self._bbox_forward(x, rois, res_feats)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_score = bbox_results['bbox_score']

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if bbox_score is None:
            bbox_score = [None] * len(cls_score)
        else:
            bbox_score = bbox_score.split(num_proposals_per_img, 0)
        rpn_score = rpn_score.split(num_proposals_per_img, 0)

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
                    bbox_score[i],
                    rpn_score[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def _bbox_forward(self, x, rois, res_feats=None, caption_features=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        det_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        vlm_roi_feats = None
        if res_feats is not None:
            vlm_roi_feats = self.vlm_roi_extractor(
                res_feats[-self.vlm_roi_extractor.num_inputs:], rois)

        if self.with_shared_head:
            det_roi_feats = self.shared_head(det_roi_feats)
        results = self.bbox_head(det_roi_feats, vlm_roi_feats)
        cls_score, bbox_pred, vlm_roi_feats, bbox_score = results
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            det_roi_feats=det_roi_feats,
            vlm_roi_feats=vlm_roi_feats,
            bbox_score=bbox_score,
        )
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      res_feats=None,
                      gt_captions=None,
                      gt_embeds=None,
                      rpn_target=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        caption_features = None
        # if gt_captions is not None:
        #     inds = [torch.randint(len(x), (1,))[0].item() for x in gt_captions]
        #     caps = [x[ind] for ind, x in zip(inds, gt_captions)]
        #     text = clip.tokenize(caps).to(x[0].device)
        #     caption_features = self.clip_model.encode_text(text).permute(1,0)
        #     caption_features = F.normalize(caption_features, dim=0, p=2)
        # assign gts and sample proposals
        if rpn_target is not None:
            proposal_index = [p[1] for p in proposal_list]
            proposal_list = [p[0] for p in proposal_list]
        else:
            proposal_index = None
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
        if self.loss_rpn_plus is not None:
            rois = bbox2roi(
                [bbox[:self.test_cfg.max_per_img] for bbox in proposal_list])
            box_feats = self.vlm_roi_extractor(res_feats[-1:], rois)
            box_feats = self.bbox_head.attnpool(box_feats)
            box_feats = F.normalize(box_feats, dim=1, p=2)
            all_embed = self.bbox_head.all_embed.type_as(box_feats)
            cls_score = box_feats @ all_embed * self.bbox_head.vlm_temperature
            labels = torch.argmax(cls_score, dim=1)
            losses['loss_rpn_plus'] = self.loss_rpn_plus(cls_score, labels)

        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x,
                sampling_results,
                gt_bboxes,
                gt_labels,
                img_metas,
                res_feats=res_feats,
                caption_features=caption_features,
                gt_embeds=gt_embeds,
                rpn_target=rpn_target,
                proposal_index=proposal_index)
            losses.update(bbox_results['loss_bbox'])
            if rpn_target is not None:
                losses.update(bbox_results['loss_rpn'])
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(
                x, sampling_results, bbox_results['det_roi_feats'], gt_masks,
                img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward_train(self,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            img_metas,
                            res_feats=None,
                            caption_features=None,
                            gt_embeds=None,
                            rpn_target=None,
                            proposal_index=None):
        """Run forward function and calculate loss for box head in training."""
        num_imgs = len(sampling_results)
        if rpn_target is not None:
            (rpn_cls_scores, rpn_bbox_preds, rpn_all_anchor_list, rpn_labels_list, rpn_label_weights_list,
                        rpn_bbox_targets_list, rpn_bbox_weights_list, rpn_num_total_samples) = rpn_target
            loss_rpn_bbox = []
            for box_pred, box_target, box_weight in zip(rpn_bbox_preds, rpn_bbox_targets_list, rpn_bbox_weights_list):
                # if self.reg_decoded_bbox:
                #     anchors = anchors.reshape(-1, 4)
                #     bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
                box_target = box_target.reshape(-1, 4)
                box_weight = box_weight.reshape(-1, 4)
                box_pred = box_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                from mmdet.models.losses import l1_loss
                loss_rpn_bbox.append(l1_loss(box_pred, box_target, box_weight, avg_factor=rpn_num_total_samples))
            loss_rpn_bbox = sum(loss_rpn_bbox)
            rpn_score_per_img, rpn_label_per_img, rpn_label_weight_per_img = [], [], []

            for i in range(num_imgs):
                rpn_score_per_img.append(torch.cat([score[i:i+1].permute(0,3,2,1).reshape(-1, 1) for  score in rpn_cls_scores]))
                rpn_label_per_img.append(torch.cat([label[i:i+1].reshape(-1) for  label in rpn_labels_list]))
                rpn_label_weight_per_img.append(torch.cat([label_weight[i:i+1].reshape(-1) for  label_weight in rpn_label_weights_list]))
            
        if caption_features is None:
            rois = bbox2roi([res.bboxes for res in sampling_results])
        # else:
        #     proposals = [res.bboxes for res in sampling_results]
        #     img_shape = [img_meta['img_shape'] for img_meta in img_metas]
        #     img_bbox = [proposals[0].new_tensor([[0, 0, w, h]]) for (h, w, _) in img_shape]
        #     proposals = [torch.cat([b, p], dim=0) for p, b in zip(proposals, img_bbox)]
        #     # proposals = self._get_top_proposals(sampling_results, img_metas)
        #     rois = bbox2roi(proposals)

        bbox_results = self._bbox_forward(x, rois, res_feats, caption_features)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  gt_bboxes,
                                                  gt_labels,
                                                  self.train_cfg,
                                                  concat=False)
        labels, label_weights, bbox_targets, bbox_weights, bbox_score_targets, bbox_score_weights  = bbox_targets
        if self.remove_novel_bbox:
            pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
            pos_level = rois[pos_inds, 0]
            pos_bbox = self.bbox_head.bbox_coder.decode(
                rois[pos_inds, 1:], bbox_targets[pos_inds])
            pos_num_per_level = pos_level.unique(return_counts=True)[1]
            pos_bbox = torch.split(pos_bbox, pos_num_per_level.tolist(), dim=0)
            pos_rois = bbox2roi(pos_bbox)
            box_feat = self.vlm_roi_extractor(res_feats[-1:], pos_rois)
            box_feat = self.bbox_head.attnpool(box_feat)
            box_feat = F.normalize(box_feat, dim=1, p=2)
            score = box_feat @ self.bbox_head.all_embed.type_as(
                box_feat) * self.bbox_head.vlm_temperature
            is_novel = (score.topk(3)[1] >=
                        len(self.bbox_head.seen_classes) - 1).any(1)
            pos_weights = bbox_weights[pos_inds].clone()
            pos_weights[is_novel] = 0.
            bbox_weights[pos_inds] = pos_weights
            pos_label_weights = label_weights[pos_inds].clone()
            pos_label_weights[is_novel] = 0.
            label_weights[pos_inds] = pos_label_weights

        if self.add_novel_bbox:
            feats = bbox_results['vlm_roi_feats']
            all_embed = self.bbox_head.all_embed.type_as(feats)

            if len(all_embed.shape) > 2: # (D, 17, 66)
                shape = all_embed.shape
                vlm_score = feats @ all_embed.reshape(shape[0], -1) * self.bbox_head.vlm_temperature
                vlm_score = vlm_score.reshape(-1, *shape[1:]).max(1)[0]
            else:
                vlm_score = feats @ all_embed * self.bbox_head.vlm_temperature
            vlm_score = vlm_score.softmax(-1)
            num_per_img = [len(s.bboxes) for s in sampling_results]
            vlm_score = torch.split(vlm_score, num_per_img)
            for i in range(num_imgs):
                pseudo_label = vlm_score[i].argmax(1)
                is_novel = (vlm_score[i][:, self.bbox_head.novel_idx] == vlm_score[i].max(1)[0][:, None]).any(1)
                is_bg = pseudo_label == self.bbox_head.num_classes
                is_novel[is_bg] = False
                if self.neg_only:
                    is_novel[:len(sampling_results[i].pos_bboxes)] = False
                is_novel = is_novel & (vlm_score[i].max(-1)[0] > 0.8)
                select_inds = is_novel.nonzero().view(-1)
                if self.select_topk_novel != 0:
                    select_inds = select_inds[:self.select_topk_novel]
                if self.select_random_k_novel != 0:
                    select_inds = select_inds[torch.randperm(len(select_inds))[:self.select_random_k_novel]]

                labels[i][select_inds] = pseudo_label[select_inds]
                label_weights[i][select_inds] = 0.2
                
                if rpn_target is not None:
                    rpn_inds = torch.cat([sampling_results[i].pos_inds, sampling_results[i].neg_inds])[select_inds]
                    rpn_inds = rpn_inds - len(gt_bboxes[i])
                    proposal_idx = proposal_index[i][rpn_inds]
                    rpn_label_per_img[i][proposal_idx] = 0
                    rpn_label_weight_per_img[i][proposal_idx] = 0.2
        
        if rpn_target is not None:
            rpn_label = torch.cat(rpn_label_per_img, 0)
            rpn_label_weight = torch.cat(rpn_label_weight_per_img, 0)
            rpn_cls_score = torch.cat(rpn_score_per_img, 0)
            from mmdet.models.losses import binary_cross_entropy
            loss_rpn_cls = binary_cross_entropy(rpn_cls_score, rpn_label, rpn_label_weight, avg_factor=rpn_num_total_samples)
        
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        bbox_score_targets = torch.cat(bbox_score_targets, 0)
        bbox_score_weights = torch.cat(bbox_score_weights, 0)

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['bbox_score'],
            rois, labels, label_weights, bbox_targets, bbox_weights,
            bbox_score_targets, bbox_score_weights,
        )
        if rpn_target is not None:
            bbox_results.update(loss_rpn=dict(loss_rpn_cls=loss_rpn_cls, loss_rpn_bbox=loss_rpn_bbox))
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
