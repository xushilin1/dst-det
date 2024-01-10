from mmdet.datasets.lvis import LVISV1Dataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

import os.path as osp
import mmcv
import json
import warnings

import itertools
import logging
import tempfile
from collections import OrderedDict
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_recalls

from collections.abc import Sequence


@DATASETS.register_module()
class LVISV1DatasetOV(LVISV1Dataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix='',
        seg_prefix=None,
        seg_suffix='.png',
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        file_client_args=dict(backend='disk'),
        seen_classes='datasets/lvis_v1_seen_classes.json',
        unseen_classes='datasets/lvis_v1_unseen_classes.json',
        all_classes='datasets/lvis_v1_all_classes.json',
    ):
        self.seen_classes = json.load(open(seen_classes))
        self.unseen_classes = json.load(open(unseen_classes))
        self.all_classes = json.load(open(all_classes))
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.seg_suffix = seg_suffix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.all_classes

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        self.coco = LVIS(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = list(self.coco.imgs.keys())
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            data_infos.append(info)
        return data_infos

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 result_file=None):
        """Evaluation in LVIS protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None):
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: LVIS style metrics.
        """

        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVISEval, LVISResults
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)

        eval_results = OrderedDict()
        # get original api
        lvis_gt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                proposal_list = []
                if isinstance(results[0], tuple):
                    proposal_list = [
                        np.concatenate(box_res, axis=0)
                        for box_res, _ in results
                    ]
                else:
                    proposal_list = [
                        np.concatenate(box_res, axis=0) for box_res in results
                    ]

                ar, ar_base, ar_novel = self.fast_eval_recall(proposal_list,
                                                              proposal_nums,
                                                              iou_thrs,
                                                              logger='silent')

                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = float(f'{ar[i]:.4f}')
                    eval_results[f'BaseAR@{num}'] = float(f'{ar_base[i]:.4f}')
                    eval_results[f'NovelAR@{num}'] = float(
                        f'{ar_novel[i]:.4f}')
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')

                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                lvis_dt = LVISResults(lvis_gt, result_files[metric])
            except IndexError:
                print_log('The testing results of the whole dataset is empty.',
                          logger=logger,
                          level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type)
            lvis_eval.params.imgIds = self.img_ids
            if metric == 'proposal':
                lvis_eval.params.useCats = 0
                lvis_eval.params.maxDets = list(proposal_nums)
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                for k, v in lvis_eval.get_results().items():
                    if k.startswith('AR'):
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[k] = val
            else:
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                lvis_results = lvis_eval.get_results()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = lvis_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    base_inds, novel_inds = [], []
                    for idx, catId in enumerate(self.cat_ids):
                        name = self.coco.load_cats([catId])[0]['name']
                        precision = precisions[:, :, idx, 0]
                        precision = precision[precision > -1]
                        ap = np.mean(precision) if precision.size else float("nan")
                        results_per_category.append(
                            (f'{name}', f'{float(ap):0.3f}'))

                        if catId in self.seen_classes:
                            base_inds.append(idx)
                        if catId in self.unseen_classes:
                            novel_inds.append(idx)

                    base_ap = precisions[:, :, base_inds, 0]
                    novel_ap = precisions[:, :, novel_inds, 0]
                    base_ap50 = precisions[0, :, base_inds, 0]
                    novel_ap50 = precisions[0, :, novel_inds, 0]

                    eval_results[f'{metric}_base_ap'] = np.mean(base_ap[base_ap > -1]) if len(base_ap[base_ap > -1]) else -1
                    eval_results[f'{metric}_novel_ap'] = np.mean(novel_ap[novel_ap > -1]) if len(novel_ap[novel_ap > -1]) else -1
                    eval_results[f'{metric}_base_ap50'] = np.mean(base_ap50[base_ap50 > -1]) if len(base_ap50[base_ap50 > -1]) else -1
                    eval_results[f'{metric}_novel_ap50'] = np.mean(novel_ap50[novel_ap50 > -1]) if len(novel_ap50[novel_ap50 > -1]) else -1

                    if result_file is not None:
                        mmcv.dump(eval_results, result_file)
                for k, v in lvis_results.items():
                    if k.startswith('AP'):
                        key = '{}_{}'.format(metric, k)
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[key] = val
                ap_summary = ' '.join([
                    '{}:{:.3f}'.format(k, float(v))
                    for k, v in lvis_results.items() if k.startswith('AP')
                ])
                eval_results['{}_mAP_copypaste'.format(metric)] = ap_summary
            lvis_eval.print_results()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        all_bboxes, base_bboxes, novel_bboxes = [], [], []
        is_base = []

        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=[self.img_ids[i]])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                all_bboxes.append(np.zeros((0, 4)))
                base_bboxes.append(np.zeros((0, 4)))
                novel_bboxes.append(np.zeros((0, 4)))
                is_base.append(np.zeros((0)))
                continue
            bboxes, base, novel = [], [], []
            is_b = []
            for ann in ann_info:
                # if ann.get('ignore', False) or ann['iscrowd']:
                #     continue
                x1, y1, w, h = ann['bbox']
                cat_name = self.coco.cats[ann['category_id']]['name']
                # assert cat_name in self.seen_classes or cat_name in self.unseen_classes
                if cat_name in self.seen_classes:
                    base.append([x1, y1, x1 + w, y1 + h])
                    is_b.append(True)
                else:
                    novel.append([x1, y1, x1 + w, y1 + h])
                    is_b.append(False)
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            base = np.array(base, dtype=np.float32)
            novel = np.array(novel, dtype=np.float32)
            is_b = np.array(is_b, dtype=np.bool)

            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            if base.shape[0] == 0:
                base = np.zeros((0, 4))
            if novel.shape[0] == 0:
                novel = np.zeros((0, 4))
            if is_b.shape[0] == 0:
                is_b = np.zeros((0))

            all_bboxes.append(bboxes)
            base_bboxes.append(base)
            novel_bboxes.append(novel)
            is_base.append(is_b)

        # recalls, base_recall, novel_recall = self.eval_recalls(
        #     all_bboxes, results, is_base, proposal_nums, iou_thrs, logger=logger)
        # ar = recalls.mean(axis=1)
        # ar_base = base_recall.mean(axis=1)
        # ar_novel = novel_recall.mean(axis=1)
        # return ar, ar_base, ar_novel

        recalls = eval_recalls(all_bboxes,
                               results,
                               proposal_nums,
                               iou_thrs,
                               logger=logger)
        ar = recalls.mean(axis=1)
        base_recall = eval_recalls(base_bboxes,
                                   results,
                                   proposal_nums,
                                   iou_thrs,
                                   logger=logger)
        ar_base = base_recall.mean(axis=1)
        novel_recall = eval_recalls(novel_bboxes,
                                    results,
                                    proposal_nums,
                                    iou_thrs,
                                    logger=logger)
        ar_novel = novel_recall.mean(axis=1)
        return ar, ar_base, ar_novel

    def set_recall_param(self, proposal_nums, iou_thrs):
        """Check proposal_nums and iou_thrs and set correct format."""
        if isinstance(proposal_nums, Sequence):
            _proposal_nums = np.array(proposal_nums)
        elif isinstance(proposal_nums, int):
            _proposal_nums = np.array([proposal_nums])
        else:
            _proposal_nums = proposal_nums

        if iou_thrs is None:
            _iou_thrs = np.array([0.5])
        elif isinstance(iou_thrs, Sequence):
            _iou_thrs = np.array(iou_thrs)
        elif isinstance(iou_thrs, float):
            _iou_thrs = np.array([iou_thrs])
        else:
            _iou_thrs = iou_thrs

        return _proposal_nums, _iou_thrs

    def _recalls(self, all_ious, is_base, proposal_nums, thrs):
        img_num = all_ious.shape[0]
        total_gt_num = sum([ious.shape[0] for ious in all_ious])

        _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
        for k, proposal_num in enumerate(proposal_nums):
            tmp_ious = np.zeros(0)
            for i in range(img_num):
                ious = all_ious[i][:, :proposal_num].copy()
                gt_ious = np.zeros((ious.shape[0]))
                if ious.size == 0:
                    tmp_ious = np.hstack((tmp_ious, gt_ious))
                    continue
                for j in range(ious.shape[0]):
                    gt_max_overlaps = ious.argmax(axis=1)
                    max_ious = ious[np.arange(0, ious.shape[0]),
                                    gt_max_overlaps]
                    gt_idx = max_ious.argmax()
                    gt_ious[j] = max_ious[gt_idx]
                    box_idx = gt_max_overlaps[gt_idx]
                    ious[gt_idx, :] = -1
                    ious[:, box_idx] = -1
                tmp_ious = np.hstack((tmp_ious, gt_ious))
            _ious[k, :] = tmp_ious

        # _ious = np.fliplr(np.sort(_ious, axis=1))
        recalls = np.zeros((proposal_nums.size, thrs.size))
        base_recall = np.zeros((proposal_nums.size, thrs.size))
        novel_recall = np.zeros((proposal_nums.size, thrs.size))

        is_base = np.concatenate(is_base, axis=0)
        is_novel = np.invert(is_base)
        for i, thr in enumerate(thrs):
            recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)
            base_recall[:, i] = (_ious[:, is_base] >= thr).sum(
                axis=1) / is_base.sum()
            novel_recall[:, i] = (_ious[:, is_novel] >= thr).sum(
                axis=1) / is_novel.sum()
        return recalls, base_recall, novel_recall

    def eval_recalls(self,
                     gts,
                     proposals,
                     is_base,
                     proposal_nums=None,
                     iou_thrs=0.5,
                     logger=None,
                     use_legacy_coordinate=False):
        img_num = len(gts)
        assert img_num == len(proposals)
        proposal_nums, iou_thrs = self.set_recall_param(
            proposal_nums, iou_thrs)
        all_ious = []
        for i in range(img_num):
            if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
                scores = proposals[i][:, 4]
                sort_idx = np.argsort(scores)[::-1]
                img_proposal = proposals[i][sort_idx, :]
            else:
                img_proposal = proposals[i]
            prop_num = min(img_proposal.shape[0], proposal_nums[-1])
            if gts[i] is None or gts[i].shape[0] == 0:
                ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
            else:
                ious = bbox_overlaps(
                    gts[i],
                    img_proposal[:prop_num, :4],
                    use_legacy_coordinate=use_legacy_coordinate)
            all_ious.append(ious)
        all_ious = np.array(all_ious)
        recalls, base_recall, novel_recall = self._recalls(
            all_ious, is_base, proposal_nums, iou_thrs)

        return recalls, base_recall, novel_recall

    def __repr__(self):
        pass