import torch 
import torch.nn.functional as F 


torch.manual_seed(0)

def attention(q, k, v):
    score = q @ k.transpose(-2, -1) # (n_heads, T, d_k) @ (n_heads, d_k, T) => (n_heads, T, T)
    score = score / q.shape[-1]**0.5 # (n_heads, T, T)
    weight = F.softmax(score, dim=-1) # (n_heads, T, T)
    out = weight @ v # (n_heads, T, T) @ (n_heads, T, d_k) => (n_heads, T, d_k) 
    return weight, out  

def multi_head_attention(emb, w_q, w_k, w_v, w_o, n_heads):
    T, dim = emb.shape 
    d_k = dim // n_heads 

    q = emb @ w_q # (T, dim) @ (dim, dim) -> (T, dim)
    k = emb @ w_k # (T, dim) @ (dim, dim) -> (T, dim)
    v = emb @ w_v # (T, dim) @ (dim, dim) -> (T, dim) 

    # split q, k, v into heads
    # (T, dim) -> (T, n_heads, d_k) -> (n_heads, T, d_k)
    q = q.view(T, n_heads, d_k).transpose(0, 1) 
    k = k.view(T, n_heads, d_k).transpose(0, 1) 
    v = v.view(T, n_heads, d_k).transpose(0, 1) 

    weight, out = attention(q, k, v)

    # (n_heads, T, d_k) -> (T, n_heads, d_k) -> (T, dim)
    out = out.transpose(0, 1).contiguous().view(T, dim)

    final = out @ w_o # (T, dim) @ (dim, dim) -> (T, dim)
    return final, weight


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    x_norm = (x - mean) / (std + eps)
    return gamma * x_norm + beta




