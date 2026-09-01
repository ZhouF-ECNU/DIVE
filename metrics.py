from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
from skimage import measure

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

# New Metric: Region-Calibrated PRO (RC-PRO) Score
def cal_rcpro_score(
    masks, 
    amaps, 
    max_step=200, 
    min_recall_threshold=0.05, 
    excess_tolerance_threshold=2.0,
    expect_fpr=0.3
):
    if masks.ndim == 2:
        masks = masks[np.newaxis, :, :]
    if amaps.ndim == 2:
        amaps = amaps[np.newaxis, :, :]

    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    
    if max_th == min_th:
        return 0.0
        
    delta = (max_th - min_th) / max_step
    if delta == 0:
        delta = 1e-8

    pro_scores, aw_rc_precision_scores, fprs = [], [], []
    pro_scores.append(0.0)
    aw_rc_precision_scores.append(1.0)
    fprs.append(0.0)
    
    labeled_masks_batch = measure.label(masks, background=0)
    gt_regions_batch = [measure.regionprops(labeled_masks_batch[i]) for i in range(masks.shape[0])]
    
    inverse_masks = 1 - masks
    total_negative_pixels = inverse_masks.sum()
    if total_negative_pixels == 0:
        total_negative_pixels = -1

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        
        # PRO (X-axis) calculation
        pro_scores_at_th = []
        for i in range(masks.shape[0]):
            gt_regions = gt_regions_batch[i]
            if not gt_regions:
                pro_scores_at_th.append(1.0 if np.sum(binary_amaps[i]) == 0 else 0.0)
                continue
            for region in gt_regions:
                tp_pixels = binary_amaps[i][region.coords[:, 0], region.coords[:, 1]].sum()
                pro_scores_at_th.append(tp_pixels / region.area)
        pro_scores.append(np.array(pro_scores_at_th).mean())

        # Region-Calibrated score (Y-axis) calculation
        batch_weighted_score_sum = 0.0
        batch_total_pred_area = 0.0
        labeled_amaps_batch = measure.label(binary_amaps, background=0)
        for i in range(masks.shape[0]):
            pred_regions = measure.regionprops(labeled_amaps_batch[i])
            if not pred_regions:
                continue
            quality_scores = _calculate_match_scores(pred_regions, gt_regions_batch[i], masks[i].shape, min_recall_threshold, excess_tolerance_threshold)
            for pred_region, score in zip(pred_regions, quality_scores):
                batch_weighted_score_sum += score * pred_region.area
                batch_total_pred_area += pred_region.area
        if batch_total_pred_area > 0:
            aw_rc_precision_scores.append(batch_weighted_score_sum / batch_total_pred_area)
        else:
            aw_rc_precision_scores.append(1.0)
            
        # FPR calculation
        if total_negative_pixels != -1:
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
            fprs.append(fp_pixels / total_negative_pixels)
        else:
            fprs.append(0.0)

    pro_scores = np.array(pro_scores)
    aw_rc_precision_scores = np.array(aw_rc_precision_scores)
    fprs = np.array(fprs)
    
    indices_to_keep = fprs <= expect_fpr
    filtered_pros = pro_scores[indices_to_keep]
    filtered_aw_rc_precisions = aw_rc_precision_scores[indices_to_keep]

    if len(filtered_pros) < 2:
        return 0.0

    sorted_indices = np.argsort(filtered_pros)
    sorted_pros = filtered_pros[sorted_indices]
    sorted_aw_rc_precisions = filtered_aw_rc_precisions[sorted_indices]
    
    if sorted_pros[0] > 0.0:
        sorted_pros = np.insert(sorted_pros, 0, 0.0)
        sorted_aw_rc_precisions = np.insert(sorted_aw_rc_precisions, 0, 1.0)
    
    auc_score = auc(sorted_pros, sorted_aw_rc_precisions)
    
    return auc_score

def _calculate_match_scores(pred_regions, gt_regions, original_mask_shape, min_recall_thresh, excess_tol_thresh):
    """
    Helper function to calculate match quality scores for a single image's predictions.
    This function's logic remains unchanged.
    """
    if not pred_regions:
        return []
    if not gt_regions:
        return [0] * len(pred_regions)

    gt_coords_map = {r.label: r.area for r in gt_regions}
    
    gt_mask = np.zeros(original_mask_shape, dtype=np.int32)
    for r in gt_regions:
        gt_mask[r.coords[:,0], r.coords[:,1]] = r.label

    scores = []
    for pred_region in pred_regions:
        pred_coords = pred_region.coords
        overlapped_gt_labels = gt_mask[pred_coords[:, 0], pred_coords[:, 1]]
        
        unique_labels, counts = np.unique(overlapped_gt_labels[overlapped_gt_labels > 0], return_counts=True)
        
        if len(unique_labels) == 0:
            scores.append(0)
            continue
        
        best_gt_label = unique_labels[np.argmax(counts)]
        overlap_area = np.max(counts)
        gt_area = gt_coords_map[best_gt_label]

        region_recall = overlap_area / gt_area
        if region_recall < min_recall_thresh:
            scores.append(0)
            continue
        
        pred_area = pred_region.area
        excess_ratio = (pred_area - overlap_area) / gt_area
        
        if excess_ratio <= excess_tol_thresh:
            score = 1.0
        else:
            score = excess_tol_thresh / excess_ratio
        
        scores.append(score)
        
    return scores


def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)

    return performance

def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'pixel-auroc':
        performance = roc_auc_score(gt.ravel(), pr.ravel())
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr)
    elif metric == 'pixel-rcpro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_rcpro_score(gt, pr)
    return performance
    