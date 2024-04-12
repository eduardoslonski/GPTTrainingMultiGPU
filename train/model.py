import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

def build_rope_cache(config, device, dtype, base=10_000):
    d = config["hidden_size"]
    theta = 1. / (base ** (torch.arange(0, d, 2)/d)).view(1, -1)
    seq_idx = torch.arange(config["context_length"]).view(-1, 1)
    m_theta = seq_idx * theta
    m_theta = m_theta.repeat(1, 2).to(device)
    cos = m_theta.cos()
    sin = m_theta.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)

def apply_rope(x, cos, sin):
    d = x.shape[-1]
    neg_half = torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1).to(x.device)
    x_rope = x * cos + neg_half * sin
    return x_rope

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.linear_1 = nn.Linear(config["hidden_size"], config["hidden_size"] * 4)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(config["hidden_size"] * 4, config["hidden_size"])

    def forward(self, x):
        # x [batch_size, context_length, hidden_size]
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config["num_attention_heads"]
        self.hidden_size = config["hidden_size"]
        self.head_dimension = config["hidden_size"] // config["num_attention_heads"]

        self.query = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.key = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.value = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.output = nn.Linear(config["hidden_size"], config["hidden_size"])

    def forward(self, x, attn_mask=None, rope_cache=None):
        # x [batch_size, context_length, hidden_size]
        batch_size, context_length, _ = x.shape

        q, k, v = self.query(x), self.key(x), self.value(x)

        cos, sin = rope_cache

        q = apply_rope(q, cos[:context_length], sin[:context_length])
        k = apply_rope(k, cos[:context_length], sin[:context_length])

        attention_mode = "flash"

        if attention_mode == "own":
            q = q.view(batch_size, context_length, self.num_attention_heads, self.head_dimension).transpose(1, 2) # [bs, n_attn_heads, context_len, head_dim]
            k = k.view(batch_size, context_length, self.num_attention_heads, self.head_dimension).transpose(1, 2) # [bs, n_attn_heads, context_len, head_dim]
            v = v.view(batch_size, context_length, self.num_attention_heads, self.head_dimension).transpose(1, 2) # [bs, n_attn_heads, context_len, head_dim]

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dimension) # [bs, n_attn_heads, context_len, context_len]

            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(attn_mask[:, :context_length, :context_length] == 0, float('-inf'))

            attn_scores = F.softmax(attn_scores, dim=-1) # [bs, n_attn_heads, context_len, context_len]
            
            attn_values = attn_scores @ v # [bs, n_attn_heads, context_len, head_dim]
            attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, context_length, -1) # [bs, contex_len, hidden_size]

        elif attention_mode == "pytorch":
            q = q.view(batch_size, context_length, self.num_attention_heads, self.head_dimension).transpose(1, 2) # [bs, n_attn_heads, context_len, head_dim]
            k = k.view(batch_size, context_length, self.num_attention_heads, self.head_dimension).transpose(1, 2) # [bs, n_attn_heads, context_len, head_dim]
            v = v.view(batch_size, context_length, self.num_attention_heads, self.head_dimension).transpose(1, 2) # [bs, n_attn_heads, context_len, head_dim]

            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            attn_values = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=True).transpose(1, 2).contiguous().view(batch_size, context_length, -1) # [bs, contex_len, hidden_size]

        elif attention_mode == "flash":
            q = q.view(batch_size, context_length, self.num_attention_heads, self.head_dimension) # [bs, context_len, n_attn_heads, head_dim]
            k = k.view(batch_size, context_length, self.num_attention_heads, self.head_dimension) # [bs, context_len, n_attn_heads, head_dim]
            v = v.view(batch_size, context_length, self.num_attention_heads, self.head_dimension) # [bs, context_len, n_attn_heads, head_dim]

            attn_values = flash_attn_func(q, k, v, dropout_p=0., causal=True).view(batch_size, context_length, -1) # [bs, contex_len, hidden_size]

        output = self.output(attn_values) # [bs, contex_len, hidden_size]

        return output

class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_norm_input = nn.LayerNorm(config["hidden_size"])
        self.attention = Attention(config)

        self.layer_norm_mlp = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)

    def forward(self, x, attn_mask, rope_cache):
        # x [batch_size, context_length, hidden_size]
        x = self.layer_norm_input(x)
        x = x + self.attention(x, attn_mask, rope_cache)
        x = self.layer_norm_mlp(x)
        x = x + self.mlp(x)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.config = config
        self.rope_cache = None

        self.transformer = nn.ModuleDict(dict(
            embedding = nn.Embedding(self.vocab_size, self.hidden_size),
            attention_blocks = nn.ModuleList([AttentionBlock(config) for _ in range(self.num_layers)])
        ))

        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        # x [batch_size, context_length]
        x = self.transformer.embedding(x) # [batch_size, context_length, hidden_size]
        attn_mask = None
        # attn_mask = torch.tril(torch.ones(x.shape[1], x.shape[1]))#.to(x.device) # [1, context_length, context_length]
        # attn_mask = attn_mask.to(x.device)
        # attn_mask = attn_mask.unsqueeze(0)
        if self.rope_cache == None:
            self.rope_cache = build_rope_cache(self.config, x.device, x.dtype)
        for layer in self.transformer.attention_blocks:
            x = layer(x, attn_mask, self.rope_cache) # [batch_size, context_length, hidden_size]
        x = self.lm_head(x) # [batch_size, context_length, vocab_size]
        return x
    
    def generate(self, input_ids, max_length, method="greedy", temperature=1.0, top_k=None, top_p=None):
        def greedy(logits, **kwargs):
            pred = torch.argmax(logits, dim=-1).unsqueeze(0)
            return pred
    
        def sampling(logits, top_k, top_p, **kwargs):
            if top_k:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
    
            if top_p:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
                sorted_indices_to_remove = cumulative_probs > top_p
                
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
    
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('Inf')
    
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            return pred
        
        method_funcs = {"greedy": greedy, "sampling": sampling}

        if method not in method_funcs:
            raise  ValueError(f"Method not found, the methods available are {list(method_funcs.keys())}")

        kwargs = {'temperature': temperature, 'top_k': top_k, 'top_p': top_p}
        original_length = input_ids.shape[-1]
        with torch.no_grad():
            self.eval()
            while (input_ids.shape[-1] - original_length) < max_length:
                input_ids = input_ids[-self.config["context_length"]:]
                logits = self(input_ids)
                logits = logits[:, -1, :] / temperature
                pred = method_funcs[method](logits, **kwargs)
                input_ids = torch.cat((input_ids, pred), dim=-1)
            self.train()

        return input_ids