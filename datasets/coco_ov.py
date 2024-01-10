import json
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from mmdet.datasets.pipelines import Compose
from collections.abc import Sequence

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

import contextlib
import io
import itertools
import logging
import warnings
from collections import OrderedDict
from mmdet.core import eval_recalls
import torch

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
# from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class CocoDatasetOV(CocoDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 caption_file=None,
                 clip_bbox_embed=None,
                 seen_classes='datasets/mscoco_seen_classes.json',
                 unseen_classes='datasets/mscoco_unseen_classes.json',
                 all_classes='datasets/mscoco_65_classes.json',
                 **kwargs):

        super().__init__(ann_file, pipeline, **kwargs)
        self.seen_classes = json.load(open(seen_classes))
        self.unseen_classes = json.load(open(unseen_classes))
        self.all_classes = json.load(open(all_classes))

        # ann_id2clip_embed
        self.bbox_embed = torch.load(
            clip_bbox_embed) if clip_bbox_embed != None else None
        if caption_file is not None:
            self.coco_caption = COCO(caption_file)
        else:
            self.coco_caption = None

        # self.CLASSES = seen_classes

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        data_info = self.data_infos[idx].copy()
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        data_info = self._parse_ann_info(self.data_infos[idx], ann_info)

        if self.coco_caption is not None:
            caption_ann_ids = self.coco_caption.get_ann_ids(img_ids=[img_id])
            caption_ann_info = self.coco_caption.load_anns(caption_ann_ids)
            # During training, randomly choose a caption as gt.
            # random_idx = np.random.randint(0, len(caption_ann_info))
            captions = [ann['caption'] for ann in caption_ann_info]
            data_info['captions'] = captions
        return data_info

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_embeds = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
            if self.bbox_embed is not None:
                gt_embeds.append(self.bbox_embed[ann['id']])

        if gt_embeds:
            gt_embeds = np.array(torch.stack(gt_embeds, dim=0),
                                 dtype=np.float32)
        else:
            gt_embeds = np.array((0, 1024), dtype=np.float32)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   embeds=gt_embeds,
                   bboxes_ignore=gt_bboxes_ignore,
                   masks=gt_masks_ann,
                   seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        # self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
        # self.cat_ids = self.known_cat_ids
        # self.cat_ids = self.unknown_cat_ids
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

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
                continue
            bboxes, base, novel = [], [], []
            is_b = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                cat_name = self.coco.cats[ann['category_id']]['name']
                assert cat_name in self.seen_classes or cat_name in self.unseen_classes
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

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
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

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log('The testing results of the whole dataset is empty.',
                          logger=logger,
                          level=logging.ERROR)
                break

            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    base_inds, novel_inds = [], []
                    for idx, catId in enumerate(self.cat_ids):

                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        name = self.coco.loadCats(catId)[0]['name']

                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        ap = np.mean(precision) if precision.size else float("nan")
                        results_per_category.append(("{}".format(name), float(ap * 100)))

                        
                        if catId in self.seen_classes:
                            base_inds.append(idx)
                        if catId in self.unseen_classes:
                            novel_inds.append(idx)
                    
                    base_ap = precisions[:, :, base_inds, 0, -1]
                    novel_ap = precisions[:, :, novel_inds, 0, -1]
                    base_ap50 = precisions[0, :, base_inds, 0, -1]
                    novel_ap50 = precisions[0, :, novel_inds, 0, -1]

                    eval_results['base_ap'] = np.mean(base_ap[base_ap > -1]) if len(base_ap[base_ap > -1]) else -1
                    eval_results['novel_ap'] = np.mean(novel_ap[novel_ap > -1]) if len(novel_ap[novel_ap > -1]) else -1
                    eval_results['base_ap50'] = np.mean(base_ap50[base_ap50 > -1]) if len(base_ap50[base_ap50 > -1]) else -1
                    eval_results['novel_ap50'] = np.mean(novel_ap50[novel_ap50 > -1]) if len(novel_ap50[novel_ap50 > -1]) else -1
                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')

        return eval_results

    def __repr__(self):
        pass



@DATASETS.register_module()
class V3DetDatasetOV(CocoDatasetOV):

    CLASSES = mmcv.list_from_file(
                'data/V3Det/annotations/category_name_13204_v3det_2023_v1.txt')
    PALETTE = None
    
    def evaluate_det_segm(self, results, result_files, coco_gt, metrics, logger=None, classwise=False, proposal_nums=(100, 300, 1000), iou_thrs=None, metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.4f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    results_per_category_seen = []
                    results_per_category_unseen = []

                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]

                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append((f'{nm["name"]}', f'{float(ap):0.3f}'))
                        
                        if catId in self.seen_classes:
                            results_per_category_seen.append(float(ap*100))
                        if catId in self.unseen_classes:
                            results_per_category_unseen.append(float(ap*100))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                # base_mAP = sum(results_per_category_seen) / len(results_per_category_seen)
                # novel_mAP = sum(results_per_category_unseen) / len(results_per_category_unseen)
                # all_mAP = sum(results_per_category_seen + results_per_category_unseen) / len(results_per_category)
                base_mAP = np.nanmean(results_per_category_seen)
                novel_mAP = np.nanmean(results_per_category_unseen)
                all_mAP = np.nanmean(results_per_category_seen + results_per_category_unseen)

                eval_results['base_mAP'] = float(f'{base_mAP:.3f}')
                eval_results['novel_mAP_nan_ig'] = float(f'{novel_mAP:.3f}')
                eval_results['all_mAP_nan_ig'] = float(f'{all_mAP:.3f}')

                results_per_category_unseen_new = []
                for i in results_per_category_unseen:
                    if i == float('nan'):
                        results_per_category_unseen_new.append(0.0)
                    else:
                        results_per_category_unseen_new.append(i)

                novel_mAP_new = sum(results_per_category_unseen_new) / len(results_per_category_unseen_new)
                eval_results['novel_mAP_nan_consider'] = float(f'{novel_mAP_new:.3f}')

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.4f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.4f} {ap[1]:.4f} {ap[2]:.4f} {ap[3]:.4f} '
                    f'{ap[4]:.4f} {ap[5]:.4f}')

        return eval_results