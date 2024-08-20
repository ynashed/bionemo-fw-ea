import functools
import time

import einops
import torch
import torch.nn.functional as F


# inputs
batch = []
cnt = 0
for i in [39, 43, 39, 38, 37, 38, 36, 38, 42, 37, 42, 41, 43, 42, 35, 37, 38, 40, 36, 39, 43, 38]:
    for j in range(i):
        batch.append(cnt)
    cnt += 1

n_head = 4
n_h = 256
p = sum([39, 43, 39, 38, 37, 38, 36, 38, 42, 37, 42, 41, 43, 42, 35, 37, 38, 40, 36, 39, 43, 38])
qkv = torch.randn((p, 3 * n_head * n_h), dtype=float, device='cuda:0')
batch = torch.tensor(batch, dtype=torch.long, device='cuda:0')


# x_norm.device, x_norm.dtype any thing before DiTe
def shared_between_layers(batch, dtype=qkv.dtype, device=qkv.device):
    particle_nums = batch.shape[0]
    _, batch_atoms = torch.unique_consecutive(batch, return_counts=True)
    n_batch = batch[-1].item() + 1

    max_b = torch.max(batch_atoms).item()

    cumsum_result = torch.cat(
        [torch.tensor([0], device=device, dtype=torch.long), torch.cumsum(max_b - batch_atoms, dim=0)[:-1]]
    )

    new_id = torch.repeat_interleave(cumsum_result, batch_atoms) + torch.arange(
        0, particle_nums, device=device, dtype=torch.long
    )

    mask = torch.tensor([[]], dtype=torch.bool, device=device)
    # max_b = batch.max().item()
    mask = torch.arange(max_b).cuda().unsqueeze(0).unsqueeze(0)  # [1, 1, max_b]
    mask = mask.expand(n_batch, max_b, max_b)  # [n_batch, max_b, max_b]
    mask = mask < batch_atoms.unsqueeze(1).unsqueeze(2)  # [n_batch, max_b, max_b]
    mask = mask & mask.transpose(1, 2)
    mask = mask.unsqueeze(1)

    return new_id, mask, n_batch, max_b


def ali_mha(qkv, batch):
    new_id, mask, n_batch, max_b = shared_between_layers(batch)
    # QKV projection
    # qkv = self.qkv_proj(x_norm)  # seq_len, num_heads*head_dim
    device = qkv.device
    # import ipdb; ipdb.set_trace()
    head_h = qkv.shape[-1]

    new_qkv = torch.zeros((n_batch * max_b, head_h), dtype=qkv.dtype, device=device)

    new_qkv[new_id, :] = qkv[:, :]

    new_qkv = new_qkv.reshape(n_batch, max_b, n_head, -1).permute((0, 2, 1, 3))  # batch x num-head x max_atoms x D
    Q, K, V = new_qkv.chunk(3, dim=-1)

    attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    # mask [1 1 num_atoms num_atoms] QKV = [1, num_heads, num_atoms, hidden//num_heads]

    attn_output = einops.rearrange(attn_output, "b h s d -> (b s) (h d)")[new_id, :]
    return attn_output


def ali_shared_mha(qkv, new_id, mask, n_batch, max_b, batch):
    # QKV projection
    # qkv = self.qkv_proj(x_norm)  # seq_len, num_heads*head_dim
    device = qkv.device
    # import ipdb; ipdb.set_trace()
    head_h = qkv.shape[-1]

    new_qkv = torch.zeros((n_batch * max_b, head_h), dtype=qkv.dtype, device=device)

    new_qkv[new_id, :] = qkv[:, :]

    new_qkv = new_qkv.reshape(n_batch, max_b, n_head, -1).permute((0, 2, 1, 3))  # batch x num-head x max_atoms x D
    Q, K, V = new_qkv.chunk(3, dim=-1)

    attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    # mask [1 1 num_atoms num_atoms] QKV = [1, num_heads, num_atoms, hidden//num_heads]

    attn_output = einops.rearrange(attn_output, "b h s d -> (b s) (h d)")[new_id, :]
    return attn_output


def og_mha(qkv, batch):
    # qkv = self.qkv_proj(x_norm)
    Q, K, V = qkv.chunk(3, dim=-1)
    # Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
    # Reshape Q, K, V to (1, seq_len, num_heads*head_dim)
    # if x.dim() == 2:
    Q = Q.unsqueeze(0)
    K = K.unsqueeze(0)
    V = V.unsqueeze(0)

    reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=n_head)
    # Reshape Q, K, V to (1, num_heads, seq_len, head_dim)
    Q, K, V = map(reshaper, (Q, K, V))

    attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(
        0
    )  # ! if float it is added as the biasbut would still need a mask s -infs?

    attn_output = F.scaled_dot_product_attention(
        Q, K, V, attn_mask=attn_mask
    )  # mask [1 1 num_atoms num_atoms] QKV = [1, num_heads, num_atoms, hidden//num_heads]
    attn_output = einops.rearrange(attn_output, "b h s d -> b s (h d)").squeeze(0)
    return attn_output


# OG_MHA
for _ in range(10):
    og_mha(qkv, batch)
torch.cuda.synchronize()

torch.cuda.synchronize()
s = time.perf_counter()
start = time.time()
for _ in range(10):
    og_mha(qkv, batch)
end = time.time()
torch.cuda.synchronize()
e = time.perf_counter()
print(f"# OG_MHA dt = {e-s} {end-start}")


# ALI_MHA_WO_SHARING

for _ in range(10):
    ali_mha(qkv, batch)
torch.cuda.synchronize()

torch.cuda.synchronize()
s = time.perf_counter()
start = time.time()
for _ in range(10):
    ali_mha(qkv, batch)
end = time.time()
torch.cuda.synchronize()
e = time.perf_counter()
print(f"# ALI_MHA_WO_SHARING dt = {e-s}{end-start}")

new_id, mask, n_batch, max_b = shared_between_layers(batch)
for _ in range(10):
    ali_shared_mha(qkv, new_id, mask, n_batch, max_b, batch)
torch.cuda.synchronize()


torch.cuda.synchronize()

s = time.perf_counter()
start = time.time()
for _ in range(10):
    # You don't need this for every layer just for the first DiTe layer
    ali_shared_mha(qkv, new_id, mask, n_batch, max_b, batch)
end = time.time()
torch.cuda.synchronize()
e = time.perf_counter()
print(f"# ALI_MHA_W_SHARING dt = {e-s}{end-start}")
