import DIVE_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import StatePromptLearner, SemanticPromptLearner
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
import clip
from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from DIVE_lib.lexicon_encode import LEXICON, encode_lexicon_features

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DIVE_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx,
        "Visual_prompt_length": 4
    }
    
    model, _ = DIVE_lib.load("ViT-L/14@336px", device=device, design_details = DIVE_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['pixel-rcpro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    # State Prompt Learner
    prompt_learner = StatePromptLearner(model.to("cpu"), DIVE_parameters)
    
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()
    
    # Initialize using the same class names as during training.
    classnames = ['woven', 'stratified', 'blotchy', 'marbled', 'perforated', 'mesh', 'fibrous', 'matted'] # DTD
    # classnames = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
    #                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood'] # MVTec
    # Semantic Prompt Learner
    semantic_prompt_learner = SemanticPromptLearner(model.to("cpu"), classnames, DIVE_parameters)
    
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)
    
    checkpoint = torch.load(args.checkpoint_path)
    
    # load prompt learner
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])

    if "semantic_prompt_learner" in checkpoint:
        semantic_prompt_learner.load_state_dict(checkpoint["semantic_prompt_learner"])
    else:
        logger.warning("semantic_prompt_learner not found in checkpoint!")

    # load projection head
    if "state_projection" in checkpoint:
        model.visual.state_projection.load_state_dict(checkpoint["state_projection"])
    else:
        logger.warning("state_projection not found in checkpoint!")
    if "semantic_projection" in checkpoint:
        try:
            model.visual.semantic_projection.load_state_dict(checkpoint["semantic_projection"])
        except Exception as e:
            logger.warning(f"Failed to load semantic_projection: {e}")
    else:
        logger.warning("semantic_projection not found in checkpoint!")
    
    # load cross-attention
    if "cross_attn_layers" in checkpoint:
        model.visual.load_state_dict(checkpoint["cross_attn_layers"], strict=False)
    else:
        logger.warning("cross_attn_layers not found in checkpoint!")

    prompt_learner.to(device)
    semantic_prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    # Encoding Lexicon Features
    combined_lexicon = LEXICON["Abnormal Lexicon"] + LEXICON["Normal Lexicon"]
    text_dict_features = encode_lexicon_features(model, combined_lexicon, device)
    text_dict_features = text_dict_features / text_dict_features.norm(dim=-1, keepdim=True)
    text_dict_features = text_dict_features.detach()

    # Pre-calculate Prompts
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    # shared_ctx
    with torch.no_grad():
        _, _, _, shared_ctx = semantic_prompt_learner()

    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, state_residual, semantic_features, semantic_residual, patch_features = model.encode_image(
                image, 
                features_list, 
                DPAM_layer=20, 
                text_dict_features=text_dict_features,
                shared_ctx=shared_ctx
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = DIVE_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = DIVE_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = similarity_map[...,1]
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = anomaly_map.sum(dim = 0)

            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)

    table_ls = []
    image_auroc_list, image_ap_list = [], []
    pixel_auroc_list, pixel_aupro_list, pixel_rcpro_list = [], [], []

    for obj in obj_list:
        table = [obj]
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()

        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.extend([str(np.round(image_auroc * 100, 1)), str(np.round(image_ap * 100, 1))])
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)

        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_rcpro = pixel_level_metrics(results, obj, "pixel-rcpro")
            table.extend([
                str(np.round(pixel_auroc * 100, 1)),
                str(np.round(pixel_aupro * 100, 1)),
                str(np.round(pixel_rcpro * 100, 1))
            ])
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_rcpro_list.append(pixel_rcpro)

        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_rcpro = pixel_level_metrics(results, obj, "pixel-rcpro")
            table.extend([
                str(np.round(pixel_auroc * 100, 1)),
                str(np.round(pixel_aupro * 100, 1)),
                str(np.round(pixel_rcpro * 100, 1)),
                str(np.round(image_auroc * 100, 1)),
                str(np.round(image_ap * 100, 1))
            ])
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_rcpro_list.append(pixel_rcpro)

        table_ls.append(table)

    if args.metrics == 'image-level':
        table_ls.append(['mean',
                         str(np.round(np.mean(image_auroc_list) * 100, 1)),
                         str(np.round(np.mean(image_ap_list) * 100, 1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")

    elif args.metrics == 'pixel-level':
        table_ls.append(['mean',
                         str(np.round(np.mean(pixel_auroc_list) * 100, 1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, 1)),
                         str(np.round(np.mean(pixel_rcpro_list) * 100, 1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_rcpro'], tablefmt="pipe")

    elif args.metrics == 'image-pixel-level':
        table_ls.append(['mean',
                         str(np.round(np.mean(pixel_auroc_list) * 100, 1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, 1)),
                         str(np.round(np.mean(pixel_rcpro_list) * 100, 1)),
                         str(np.round(np.mean(image_auroc_list) * 100, 1)),
                         str(np.round(np.mean(image_ap_list) * 100, 1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_rcpro', 'image_auroc', 'image_ap'], tablefmt="pipe")

    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DIVE", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)