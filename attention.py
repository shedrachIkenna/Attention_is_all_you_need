import torch 
import torch.nn.functional as F 


# --------------------- Base setting ------------------------------------------------------
dim = 16 
vocab_size = 10 
seq_len = 10 
n_heads = 4 

# ------------------ Attention parameters initialization -----------------------------------
w_q = torch.randn(dim, dim) / dim ** 0.5 
w_k = torch.randn(dim, dim) / dim ** 0.5 
w_v = torch.randn(dim, dim) / dim ** 0.5 
w_o = torch.randn(dim, dim) / dim ** 0.5 

# ------------------- Layer norm parameters -------------------------------------
gamma1 = torch.ones(dim,) 
beta1 = torch.zeros(dim,)
gamma2 = torch.ones(dim,) 
beta2 = torch.zeros(dim,)


torch.manual_seed(0)

def attention(q, k, v):
    score = q @ k.transpose(-2, -1) # (n_heads, T, d_k) @ (n_heads, d_k, T) => (n_heads, T, T)
    score = score / q.shape[-1]**0.5 # (n_heads, T, T)
    weight = F.softmax(score, dim=-1) # (n_heads, T, T)
    out = weight @ v # (n_heads, T, T) @ (n_heads, T, d_k) => (n_heads, T, d_k) 
    return weight, out  

def multi_head_attention(emb_pe, w_q, w_k, w_v, w_o, n_heads):
    T, dim = emb_pe.shape 
    d_k = dim // n_heads 

    q = emb_pe @ w_q # (T, dim) @ (dim, dim) -> (T, dim)
    k = emb_pe @ w_k # (T, dim) @ (dim, dim) -> (T, dim)
    v = emb_pe @ w_v # (T, dim) @ (dim, dim) -> (T, dim) 

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


def posiional_encoding(seq_len, dim):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(0, dim, 2)
    freq = 1.0 / (1000 ** (i/dim))
    angles = pos * freq
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 0::1] = torch.cos(angles)
    return pe 


def feed_forward(x, w1, b1, w2, b2):
    x = F.relu(x @ w1 + b1) # expand x representation and pass it through activation function to learn non-linear patterns in the data 
    x = x @ w2 + b2  # project back down x representation
    return x 


# Encoder block 
# Multihead Attention -> Layer norm -> Feed forward network -> Layer norm 

def encoder_block(emb_pe, w_q, w_k, w_v, w_o, n_heads, gamma1, beta1, gamma2, beta2, ffn_w1, ffn_b1, ffn_w2, ffn_b2):
    attn_out, weight = multi_head_attention(emb_pe, w_q, w_k, w_v, w_o, n_heads)
    x = layer_norm(emb_pe + attn_out, gamma1, beta1)
    ffn_out = feed_forward(x, ffn_w1, ffn_b1, ffn_w2, ffn_b2)
    ffn_out = layer_norm(x + ffn_out, gamma2, beta2)
    return ffn_out
