"""Low-VRAM inference helpers for MiniFold.

Two memory sinks dominate MiniFold inference on a small card:

1. The frozen ESM-2 3B encoder. fair-esm stores it as fp16 but instantiates it
   in fp32 (10.6 GiB), which does not fit on an 8 GiB GPU.
2. The stacked attention maps. MiniFold feeds every ESM attention map into the
   pair representation, so `ESM2.forward(need_head_weights=True)` materialises
   a B x L x L x (36*40) tensor: 4.7 GiB at L=1280 even in bf16.

`offload_lm_layers` addresses the first by keeping the transformer layers in
host RAM and streaming them in one at a time. `forward_streaming` addresses the
second by folding each layer's attention map into the pair projection as soon as
it is produced. The projection `fc_z[0]` is linear in the stacked-head axis, so
the chunked sum is algebraically identical to the stock path.
"""

import torch
import torch.nn.functional as F

from minifold.utils.tensor_utils import tensor_tree_map
from minifold.utils.feats import atom14_to_atom37


def offload_lm_layers(lm, device):
    """Stream ESM transformer layers from pinned host RAM, one at a time.

    The CPU copy stays authoritative: the forward hook drops the GPU replica
    rather than copying weights back, so each prediction costs one host->device
    pass over the encoder (~5.3 GiB in bf16) and no device->host traffic.
    """
    for layer in lm.layers:
        layer.to("cpu")
        masters = []
        for p in layer.parameters():
            p.data = p.data.pin_memory()
            masters.append(p.data)
        layer._cpu_masters = masters

    def pre_hook(module, _args):
        for p, cpu in zip(module.parameters(), module._cpu_masters):
            p.data = cpu.to(device, non_blocking=True)

    def post_hook(module, _args, output):
        for p, cpu in zip(module.parameters(), module._cpu_masters):
            p.data = cpu
        return output

    for layer in lm.layers:
        layer.register_forward_pre_hook(pre_hook)
        layer.register_forward_hook(post_hook)


def _lm_forward_streaming(lm, tokens, fc_z_in):
    """ESM2.forward specialised for MiniFold: returns (final repr, pair proj).

    Mirrors minifold/utils/esm.py::ESM2.forward, except that the per-layer
    attention map is projected through `fc_z_in` (the Linear(1440, 128) that
    opens MiniFold's fc_z) and accumulated, instead of being stored.
    """
    assert tokens.ndim == 2
    padding_mask = tokens.eq(lm.padding_idx)
    x = lm.embed_scale * lm.embed_tokens(tokens)

    if lm.token_dropout:
        x.masked_fill_((tokens == lm.mask_idx).unsqueeze(-1), 0.0)
        mask_ratio_train = 0.15 * 0.8
        src_lengths = (~padding_mask).sum(-1)
        mask_ratio_observed = (tokens == lm.mask_idx).sum(-1).to(x.dtype) / src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    B, T = tokens.shape
    heads = lm.attention_heads
    W, bias = fc_z_in.weight, fc_z_in.bias
    acc = torch.zeros(B, T, T, W.shape[0], device=x.device, dtype=torch.float32)

    x = x.transpose(0, 1)  # (B, T, E) -> (T, B, E)
    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(lm.layers):
        x, attn = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=True)
        # (H, B, T, T) -> (B, T, T, H), projected and discarded immediately
        attn = attn.permute(1, 2, 3, 0)
        cols = slice(layer_idx * heads, (layer_idx + 1) * heads)
        acc += F.linear(attn, W[:, cols].to(attn.dtype)).float()
        del attn

    x = lm.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) -> (B, T, E)
    return x, acc.add_(bias.float())


def forward_streaming(model, batch, num_recycling=0):
    """MiniFoldModel.forward with the attention stack consumed layer-by-layer."""
    with torch.autocast("cuda", enabled=False):
        s_s, s_z = _lm_forward_streaming(model.lm, batch["seq"], model.fc_z[0])

    s_s = model.fc_s(s_s)
    s_z = model.fc_z[2](F.relu(s_z.to(s_s.dtype)))

    preds, s_z = model.fold(s_s, s_z, mask=batch["mask"], num_recycling=num_recycling)
    r_dict = {"preds": preds, "pair": s_z}

    mask = batch["mask"]
    pair_mask = mask[:, None, :] * mask[:, :, None]
    r_dict["single"] = model.sz_project(s_z, s_s, pair_mask)
    feats = tensor_tree_map(lambda t: t[..., 0], batch["batch_of"])
    r_dict["sm"] = model.structure_module(
        s=r_dict["single"],
        z=r_dict["pair"],
        aatype=feats["aatype"],
        mask=feats["seq_mask"].to(dtype=r_dict["single"].dtype),
    )
    r_dict["final_atom_positions"] = atom14_to_atom37(r_dict["sm"]["positions"][-1], feats)
    r_dict["final_atom_mask"] = feats["atom37_atom_exists"]
    r_dict["final_affine_tensor"] = r_dict["sm"]["frames"][-1]
    r_dict.update(model.aux_heads(r_dict))
    return r_dict
