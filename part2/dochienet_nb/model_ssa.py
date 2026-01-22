
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def sinusoidal_position_encoding(pos, dim):
    pos = pos.float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=pos.device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(pos.size(0), dim, device=pos.device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# ====== KEY SPEED FIX: cache attention masks ======
_ATTN_MASK_CACHE = {}

def build_block_diag_attn_mask(seq_len: int, window: int, device):
    """
    bool mask [L, L], True means masked (NOT allowed to attend)
    cached by (L, window, device_type)
    """
    key = (seq_len, window, device.type)
    m = _ATTN_MASK_CACHE.get(key, None)
    if m is not None:
        return m
    win_id = torch.arange(seq_len, device=device) // window
    mask = ~(win_id.unsqueeze(0) == win_id.unsqueeze(1))  # True if different window => masked
    _ATTN_MASK_CACHE[key] = mask
    return mask

class SSALayer(nn.Module):
    def __init__(self, hidden_size=768, heads=8, ff=2048, dropout=0.1, window=48, shift=0):
        super().__init__()
        self.window = window
        self.shift = shift

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, hidden_size),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, H]
        B, L, H = x.shape

        if self.shift != 0:
            x = torch.roll(x, shifts=-self.shift, dims=1)

        attn_mask = build_block_diag_attn_mask(L, self.window, x.device)  # cached

        q = self.norm1(x)
        y, _ = self.attn(q, q, q, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop1(y)

        if self.shift != 0:
            x = torch.roll(x, shifts=self.shift, dims=1)

        z = self.norm2(x)
        z = self.ffn(z)
        x = x + self.drop2(z)
        return x

class SSADecoder(nn.Module):
    def __init__(self, hidden_size=768, heads=8, ff=2048, dropout=0.1, window=48, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            shift = 0 if (i % 2 == 0) else (window // 2)
            self.layers.append(SSALayer(hidden_size, heads, ff, dropout, window, shift))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DHFormerWithSSA(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 768,
        root_id: int = 0,
        max_inner_pos: int = 2048,
        page_pe_dim: int = 128,
        ssa_layers: int = 2,
        ssa_heads: int = 8,
        ssa_ff: int = 2048,
        ssa_dropout: float = 0.1,
        window: int = 48,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.root_id = root_id

        self.root_emb = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.root_emb, mean=0.0, std=0.02)

        self.inner_emb = nn.Embedding(max_inner_pos, hidden_size)
        self.page_proj = nn.Linear(page_pe_dim, hidden_size)
        self.page_pe_dim = page_pe_dim

        self.elem_decoder = SSADecoder(
            hidden_size=hidden_size,
            heads=ssa_heads,
            ff=ssa_ff,
            dropout=ssa_dropout,
            window=window,
            layers=ssa_layers
        )

        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)

    def _encode_one_chunk(self, c):
        input_ids = torch.tensor(c["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.tensor(c["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
        bbox = torch.tensor(c["bbox"], dtype=torch.long, device=device).unsqueeze(0)

        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, return_dict=True)
        hidden = o.last_hidden_state.squeeze(0)  # [T,H]

        page_ids = torch.tensor(c["page_id"], dtype=torch.long, device=device)
        inner_pos = torch.tensor(c["inner_pos"], dtype=torch.long, device=device)

        pe = sinusoidal_position_encoding(page_ids, self.page_pe_dim)
        page_e = self.page_proj(pe)

        inner_pos = torch.clamp(inner_pos, 0, self.inner_emb.num_embeddings - 1)
        inner_e = self.inner_emb(inner_pos)

        return hidden + page_e + inner_e

    def forward(self, chunks, doc_elem_ids, doc_elem_positions, parent_map):
        enc_outs = [self._encode_one_chunk(c) for c in chunks]  # list [T,H]

        elem_emb = {}
        for eid, (chunk_idx, tok_idx) in doc_elem_positions.items():
            elem_emb[eid] = enc_outs[chunk_idx][tok_idx]

        elem_seq = torch.stack([self.root_emb] + [elem_emb[eid] for eid in doc_elem_ids], dim=0)  # [L,H]
        elem_seq = elem_seq.unsqueeze(0)  # [1,L,H]

        elem_ctx = self.elem_decoder(elem_seq).squeeze(0)  # [L,H]

        decoder_elem_ids = [self.root_id] + doc_elem_ids
        id2idx = {pid: i for i, pid in enumerate(decoder_elem_ids)}
        parent_target = torch.tensor([id2idx[parent_map[eid]] for eid in doc_elem_ids],
                                     dtype=torch.long, device=device)

        child_emb = elem_ctx[1:]
        parent_emb = elem_ctx

        logits = []
        for i in range(child_emb.size(0)):
            c = child_emb[i].unsqueeze(0).repeat(parent_emb.size(0), 1)
            logits.append(self.bilinear(c, parent_emb).squeeze(-1))
        logits = torch.stack(logits, dim=0)

        loss = F.cross_entropy(logits, parent_target)
        return loss, logits

# instantiate SSA model
MODEL_NAME = "microsoft/layoutlmv3-base"
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)

model_ssa = DHFormerWithSSA(
    encoder=encoder,
    hidden_size=768,
    root_id=0,
    window=48,
    ssa_layers=2,
    ssa_heads=8,
    ssa_ff=2048
).to(device)

print("model_ssa ready on", device)


