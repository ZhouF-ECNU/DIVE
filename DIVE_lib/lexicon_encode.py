LEXICON = {
    "Abnormal Lexicon": [
        "a photo of an object with a crack",
        "a photo of an object with a fracture",
        "a photo of an object with a hole",
        "a photo of an object with a pit",
        "a photo of an object with a void",
        "a photo of an object with a gap",
        "a photo of an object with a tear",
        "a photo of an object with a chip",
        "a photo of an object with a blister",
        "a photo of an object with a bump",
        "a photo of an object with a bulge",
        "a photo of an object with a wrinkle",
        "a photo of an object with a fold",
        "a photo of an object with a crease",
        "a photo of an object with a deformation",
        "a photo of an object with a distortion",
        "a photo of an object with a warp",
        "a photo of an object with a bend",
        "a photo of an object with a twist",
        "a photo of an object with a buckle",
        "a photo of an object with a misalignment",
        "a photo of an object with a delamination",
        "a photo of an object with a separation",
        "a photo of an object with a break",
        "a photo of an object with a split",
        "a photo of an object with a leak",
        "a photo of an object with a stain",
        "a photo of an object with a smudge",
        "a photo of an object with a smear",
        "a photo of an object with a spot",
        "a photo of an object with a speck",
        "a photo of an object with a patch",
        "a photo of an object with a blotch",
        "a photo of an object with a mark",
        "a photo of an object with a burn",
        "a photo of an object with a scorch",
        "a photo of an object with a residue",
        "a photo of an object with a deposit",
        "a photo of an object with a contamination",
        "a photo of an object with a foreign particle",
        "a photo of an object with a dust particle",
        "a photo of an object with an oil stain",
        "a photo of an object with a fingerprint",
        "a photo of an object with a discoloration",
        "a photo of an object with fading",
        "a photo of an object with bleaching",
        "a photo of an object with oxidation",
        "a photo of an object with corrosion",
        "a photo of an object with rust",
        "a photo of an object with erosion",
        "a photo of an object with abrasion",
        "a photo of an object with roughness",
        "a photo of an object with pitting",
        "a photo of an object with a microcrack",
        "a photo of an object with a scrape",
        "a photo of an object with a scratch line",
        "a photo of an object with a fracture line",
        "a photo of an object with a surface defect",
        "a photo of an object with surface unevenness",
        "a photo of an object with a texture irregularity",
        "a photo of an object with a texture anomaly",
        "a photo of an object with a color shift",
        "a photo of an object with a color inconsistency",
        "a photo of an object with uneven brightness",
        "a photo of an object with uneven contrast",
        "a photo of an object with a shadow artifact",
        "a photo of an object with a reflection artifact",
        "a photo of an object with blur",
        "a photo of an object with noise",
        "a photo of an object with an artifact",
        "a photo of an object with an edge defect",
        "a photo of an object with a corner chip",
        "a photo of an object with rim damage",
        "a photo of an object with peeling",
        "a photo of an object with flaking",
        "a photo of an object with bubbling",
        "a photo of an object with void inclusion",
        "a photo of an object with foreign inclusion",
        "a photo of an object with an impurity",
        "a photo of an object with a residual stress mark",
        "a photo of an object with a flow line",
        "a photo of an object with a weld line",
        "a photo of an object with a joint defect",
        "a photo of an object with a micro void",
        "a photo of an object with a grain boundary crack",
        "a photo of an object with a delustered area",
        "a photo of an object with a pattern distortion",
        "a photo of an object with surface contamination",
        "a photo of an object with a rough patch",
        "a photo of an object with non-uniform coating",
        "a photo of an object with incomplete coverage",
        "a photo of an object with a color spot",
        "a photo of an object with a texture mismatch",
        "a photo of an object with a local defect",
        "a photo of an object with a structural defect",
        "a photo of an object with a material defect",
        "a photo of an object with a surface anomaly",
        "a photo of an object with a visual irregularity"
    ],
   "Normal Lexicon": [
        "a photo of an object with a flawless surface",
        "a photo of an object with perfect condition",
        "a photo of an object with smooth appearance",
        "a photo of an object with uniform texture",
        "a photo of an object with consistent color",
        "a photo of an object with even tone",
        "a photo of an object with a glossy finish",
        "a photo of an object with a clean surface",
        "a photo of an object with an intact structure",
        "a photo of an object with a stable form",
        "a photo of an object with a regular pattern",
        "a photo of an object with balanced shading",
        "a photo of an object with a homogeneous area",
        "a photo of an object with fine grain",
        "a photo of an object with a clear boundary",
        "a photo of an object with a neat surface",
        "a photo of an object with continuous texture",
        "a photo of an object with a well-formed edge",
        "a photo of an object with a well-aligned part",
        "a photo of an object with consistent gloss",
        "a photo of an object with a flat surface",
        "a photo of an object with solid integrity",
        "a photo of an object with a precise shape",
        "a photo of an object with uniform illumination",
        "a photo of an object with stable texture",
        "a photo of an object with a coherent structure",
        "a photo of an object with a smooth gradient",
        "a photo of an object with natural color",
        "a photo of an object with even coating",
        "a photo of an object with well-preserved material",
        "a photo of an object with an unblemished finish",
        "a photo of an object with sound condition",
        "a photo of an object with a defect-free area",
        "a photo of an object with perfect alignment",
        "a photo of an object with a clean appearance"
    ]
}

import torch
import clip

def encode_lexicon_features(model, text_list, device):
    text_embeds_list = []
    
    with torch.no_grad():
        for text in text_list:
            # Tokenize
            text_tokens = clip.tokenize([text]).to(device)  # [1, 77]
            
            x = model.token_embedding(text_tokens) # [1, 77, 768]
            x = x + model.positional_embedding[:77, :]
            x = x.permute(1, 0, 2)  # [77, 1, 768] - NLD -> LND
            
            for i, block in enumerate(model.transformer.resblocks):
                ln_1_out = block.ln_1(x)

                attn_out = block.attn(ln_1_out, ln_1_out, ln_1_out, need_weights=False)[0]
                
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))
            
            x = x.permute(1, 0, 2)  # [1, 77, 768] - LND -> NLD
            x = model.ln_final(x)
            
            eot_idx = text_tokens.argmax(dim=-1) # [1]
            text_embed = x[torch.arange(x.shape[0]), eot_idx] @ model.text_projection  # [1, 768]
            
            text_embeds_list.append(text_embed)
        
        text_embeds = torch.cat(text_embeds_list, dim=0)  # [N_texts, 768]
    
    return text_embeds