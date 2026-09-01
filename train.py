import DIVE_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import StatePromptLearner, SemanticPromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize, ModelEMA
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from DIVE_lib.lexicon_encode import LEXICON, encode_lexicon_features


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DIVE_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx,
        "Visual_prompt_length": 4
    }

    model, _ = DIVE_lib.load("ViT-L/14@336px", device=device, design_details = DIVE_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    classnames = train_data.obj_list

    # State Prompt Learner
    prompt_learner = StatePromptLearner(model.to("cpu"), DIVE_parameters)
    prompt_learner.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()

    # Semantic Prompt Learner
    semantic_prompt_learner = SemanticPromptLearner(model.to("cpu"), classnames, DIVE_parameters)
    semantic_prompt_learner.to(device)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)

    model.to(device)

    for param in model.visual.parameters():
        param.requires_grad = False

    # unfreeze the projection heads
    if hasattr(model.visual, 'state_projection'):
        for param in model.visual.state_projection.parameters():
            param.requires_grad = True

    if hasattr(model.visual, 'semantic_projection'):
        for param in model.visual.semantic_projection.parameters():
            param.requires_grad = True
    
    # encoding lexicon features
    combined_lexicon = LEXICON["Abnormal Lexicon"] + LEXICON["Normal Lexicon"]
    text_dict_features = encode_lexicon_features(model, combined_lexicon, device)
    text_dict_features = text_dict_features / text_dict_features.norm(dim=-1, keepdim=True)
    text_dict_features = text_dict_features.detach()

    # optimizer
    params_prompt = list(prompt_learner.parameters())
    params_prompt.extend(list(semantic_prompt_learner.parameters()))
    params_visual = []

    # projection heads
    if hasattr(model.visual, 'state_projection'):
        params_visual.extend(list(model.visual.state_projection.parameters()))
    if hasattr(model.visual, 'semantic_projection'):
        params_visual.extend(list(model.visual.semantic_projection.parameters()))
    
    # cross attention networks
    cross_attn_params = []
    for name, param in model.visual.named_parameters():
        if "cross_attn_layers" in name:
            param.requires_grad = True
            cross_attn_params.append(param)
    if len(cross_attn_params) > 0:
        params_visual.extend(cross_attn_params)
    else:
        logger.warning("No Cross-Attention parameters found to optimize! Check model definition.")

    optimizer = torch.optim.Adam([
        {'params': params_prompt, 'lr': args.learning_rate},
        {'params': params_visual, 'lr': args.learning_rate * 0.05}
    ], betas=(0.5, 0.999))
    ema = ModelEMA(model, decay=0.999)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # weighting coefficient
    lambda_semantic = 0.5
    lambda_decoupling = 0.1
    
    model.eval()
    prompt_learner.train()
    semantic_prompt_learner.train()
    
    for epoch in tqdm(range(args.epoch)):
        model.eval() 
        prompt_learner.train()
        semantic_prompt_learner.train()

        loss_list = []
        image_loss_list = []
        semantic_loss_list = []
        decoupling_loss_list = []

        for step, items in enumerate(tqdm(train_dataloader)):
            image = items['img'].to(device)
            label =  items['anomaly']
            cls_id = items['cls_id'].to(device) if 'cls_id' in items else None

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            semantic_prompts, semantic_tokenized_prompts, semantic_compound_prompts_text, shared_ctx = semantic_prompt_learner(cls_id=None)
            shared_ctx_ = shared_ctx.detach()
            
            state_features, state_residual, semantic_features, semantic_residual, patch_features = model.encode_image(
                image, 
                args.features_list, 
                DPAM_layer = 20, 
                text_dict_features = text_dict_features,
                shared_ctx = shared_ctx_
            )
            state_features = state_features / state_features.norm(dim=-1, keepdim=True)
            semantic_features = semantic_features / semantic_features.norm(dim=-1, keepdim=True)
                    
            # state loss
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            text_features_state = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features_state = torch.stack(torch.chunk(text_features_state, dim = 0, chunks = 2), dim = 1)
            text_features_state = text_features_state / text_features_state.norm(dim=-1, keepdim=True)
            
            text_probs = state_features.unsqueeze(1) @ text_features_state.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())

            # semantic loss
            text_features_semantic = model.encode_text_learn(semantic_prompts, semantic_tokenized_prompts, semantic_compound_prompts_text).float()
            text_features_semantic = text_features_semantic / text_features_semantic.norm(dim=-1, keepdim=True)

            n_cls = len(classnames)
            template_num = semantic_prompt_learner.template_num
            text_features_semantic = text_features_semantic.reshape(n_cls, template_num, -1)
            text_features_semantic = text_features_semantic.mean(dim=1)

            logits_semantic = semantic_features @ text_features_semantic.t() / 0.07
            if cls_id is not None:
                semantic_loss = F.cross_entropy(logits_semantic, cls_id)
            else:
                semantic_loss = torch.tensor(0.0).to(device)
            semantic_loss_list.append(semantic_loss.item())

            # Regularization constraint
            visual_prompt_reg_loss = 0.01 * (shared_ctx ** 2).mean()

            # disentanglement loss
            cos_sim = (state_residual * semantic_residual).sum(dim=-1)
            loss_decoupling = (cos_sim ** 2).mean()
            decoupling_loss_list.append(loss_decoupling.item())

            # segmentation loss
            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = DIVE_lib.compute_similarity(patch_feature, text_features_state[0])
                    similarity_map = DIVE_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss_seg = 0
            for i in range(len(similarity_map_list)):
                loss_seg += loss_focal(similarity_map_list[i], gt)
                loss_seg += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss_seg += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            total_loss = image_loss + lambda_semantic * semantic_loss + lambda_decoupling * loss_decoupling + loss_seg + visual_prompt_reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            ema.update()
            loss_list.append(total_loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(
                epoch + 1, args.epoch, 
                np.mean(loss_list)
            ))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            ema.apply_shadow()
            cross_attn_state = {k: v.cpu() for k, v in model.visual.state_dict().items() if "cross_attn_layers" in k}
            
            torch.save({
                "prompt_learner": prompt_learner.state_dict(),
                "semantic_prompt_learner": semantic_prompt_learner.state_dict(),
                "cross_attn_layers": cross_attn_state,
                "state_projection": model.visual.state_projection.state_dict() if hasattr(model.visual, 'state_projection') else None,
                "semantic_projection": model.visual.semantic_projection.state_dict() if hasattr(model.visual, 'semantic_projection') else None,
            }, ckp_path)
            ema.restore()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DIVE", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)