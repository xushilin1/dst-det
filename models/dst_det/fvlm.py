from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.nn as nn

@DETECTORS.register_module()
class Fvlm(TwoStageDetector):

    def __init__(self, backbone_name=None, use_res_feature=False,
                 **kwargs):
        super(Fvlm, self).__init__(**kwargs)
        self.backbone_name = backbone_name
        self.use_res_feature = use_res_feature
        if backbone_name is None or 'convnext' in backbone_name:
            self.roi_head.bbox_head.attnpool = self.backbone.attnpool
        if self.rpn_head.__class__.__name__ == 'ModifiedRPNHead':
            self.rpn_head.attnpool = self.backbone.attnpool

    def init_weights(self):
        super(Fvlm, self).init_weights()
        if self.backbone_name is not None:
            import clip
            clip_model, _ = clip.load(self.backbone_name, device='cpu')
            #         # model.visual.a
            state_dict = clip_model.visual.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if 'attnpool' not in k}
            self.backbone.load_state_dict(state_dict)

            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

            self.roi_head.bbox_head.attnpool.load_state_dict(clip_model.visual.attnpool.state_dict())

            self.roi_head.bbox_head.attnpool.eval()
            for param in self.roi_head.bbox_head.attnpool.parameters():
                param.requires_grad = False
                
            if hasattr(self.rpn_head, 'attnpool') and self.rpn_head.attnpool is not None:
                self.rpn_head.attnpool.load_state_dict(clip_model.visual.attnpool.state_dict())
                self.rpn_head.attnpool.eval()
                for param in self.rpn_head.attnpool.parameters():
                    param.requires_grad = False

            del clip_model

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        res_feats = self.backbone(img)
        if self.with_neck:
            x = self.neck(res_feats)
        else:
            x = res_feats
        if hasattr(self.backbone, 'model_name'):
            res_feats[-1] = self.backbone.clip_model.visual.trunk.norm_pre(res_feats[-1])
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        res = self.roi_head.simple_test(x,
                                        proposal_list,
                                        img_metas,
                                        res_feats=res_feats[-1:],
                                        rescale=rescale)
        return res


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_captions=None,
                      gt_embeds=None,
                      **kwargs):
        # x = self.extract_feat(img)
        res_feats = self.backbone(img)
        if self.with_neck:
            x = self.neck(res_feats)

        if self.use_res_feature:
            res_feats = res_feats[-1:]
        else:
            res_feats = None

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            if hasattr(self.backbone, 'model_name') and self.use_res_feature: # convnext
                res_feats[-1] = self.backbone.clip_model.visual.trunk.norm_pre(res_feats[-1])
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if hasattr(self.rpn_head, 'attnpool'):
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    res_feats=res_feats[-1:],
                    **kwargs)
            else:   
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
            if  not hasattr(self.rpn_head, 'no_rpn_loss') or not self.rpn_head.no_rpn_loss:
                losses.update(rpn_losses)
                rpn_losses = None
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 res_feats=res_feats,
                                                 gt_captions=gt_captions,
                                                 gt_embeds=gt_embeds,
                                                 rpn_target=rpn_losses,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses


    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if name.startswith('backbone'):
                module.train(False)
            else:
                module.train(mode)
        if hasattr(self.roi_head.bbox_head, 'attnpool') and self.roi_head.bbox_head.attnpool:
            if isinstance(self.roi_head.bbox_head.attnpool, nn.Module):
                for p in self.roi_head.bbox_head.attnpool.parameters():
                    p.requires_grad = False
                self.roi_head.bbox_head.attnpool.eval()
        if hasattr(self.rpn_head, 'attnpool') and self.rpn_head.attnpool is not None:
            if isinstance(self.rpn_head.attnpool, nn.Module):
                for p in self.rpn_head.attnpool.parameters():
                    p.requires_grad = False
                self.rpn_head.attnpool.eval()
