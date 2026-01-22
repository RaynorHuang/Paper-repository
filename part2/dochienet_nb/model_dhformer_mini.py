from . import chunking_demo
from . import label_parsing
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class DHFormerMini(nn.Module):
    """
    Minimal DHFormer-like wrapper:
      - real text-layout encoder (LayoutLMv3/GeoLayoutLM/etc.)
      - element pooling by first-token
      - root learnable embedding
      - bilinear parent scorer
    Input format:
      chunks: list of dict with token-level fields
      doc_elem_ids: list of element ids (no root)
      doc_elem_positions: dict elem_id -> (chunk_idx, token_idx)
      parent_map: dict child_elem_id -> parent_elem_id (root=0)
    """
    def __init__(self, encoder: nn.Module, hidden_size: int = 768, root_id: int = 0):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.root_id = root_id

        self.root_emb = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.root_emb, mean=0.0, std=0.02)

        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)

    def encode_chunks(self, chunks):
        
        outs = []
        for c in chunks:
            input_ids = torch.tensor(c["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(c["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
            bbox = torch.tensor(c["bbox"], dtype=torch.long, device=device).unsqueeze(0)

            o = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                return_dict=True
            )
            outs.append(o.last_hidden_state.squeeze(0))
        return outs

    def forward(self, chunks, doc_elem_ids, doc_elem_positions, parent_map):
        
        self.encoder.eval() 
        with torch.set_grad_enabled(self.training):
            enc_outs = self.encode_chunks(chunks)

        
        elem_emb = {}
        for eid, (chunk_idx, tok_idx) in doc_elem_positions.items():
            elem_emb[eid] = enc_outs[chunk_idx][tok_idx]  # [H]

        
        decoder_inputs = torch.stack(
            [self.root_emb.to(device)] + [elem_emb[eid] for eid in doc_elem_ids],
            dim=0
        )  

    
        decoder_elem_ids = [self.root_id] + doc_elem_ids
        id2idx = {pid: i for i, pid in enumerate(decoder_elem_ids)}
        parent_target = torch.tensor(
            [id2idx[parent_map[eid]] for eid in doc_elem_ids],
            dtype=torch.long,
            device=device
        ) 

      
        child_emb = decoder_inputs[1:]   
        parent_emb = decoder_inputs      

        logits = []
        for i in range(child_emb.size(0)):
            c = child_emb[i].unsqueeze(0).repeat(parent_emb.size(0), 1)
            score = self.bilinear(c, parent_emb).squeeze(-1)
            logits.append(score)
        logits = torch.stack(logits, dim=0)

        loss = F.cross_entropy(logits, parent_target)
        return loss, logits


MODEL_NAME = "microsoft/layoutlmv3-base"
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)

model = DHFormerMini(encoder=encoder, hidden_size=768, root_id=0).to(device)
model.eval()

loss, logits = model(
    chunking_demo.chunks,
    chunking_demo.doc_elem_ids,
    chunking_demo.doc_elem_positions,
    chunking_demo.parent_map,
)



print("loss:", float(loss))
print("logits shape:", tuple(logits.shape))


