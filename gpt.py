import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW

block_size = 8
batch_size = 32
max_iter = 3000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iter = 200


text = ''
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}
encode = lambda text: [stoi[s] for s in text]
decode = lambda encoded: "".join([itos[i] for i in encoded])

data = torch.tensor(encode(text))
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    xb = torch.stack([train_data[i:i+block_size] for i in ix])
    yb = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return xb.to(device),yb.to(device)

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    def forward(self, idx,target=None):
        logits = self.token_embedding_table(idx)
        if target == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        return logits, loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iter)
        for iter in range(eval_iter):
            xb,yb = get_batch(split)
            _,loss = model(xb,yb)
            losses[iter] = loss
        out[split] = losses.mean()
    model.train()
    return out



    

model = BigramLanguageModel(len(vocab))
model = model.to(device)
optim = AdamW(model.parameters())
batch_size = 32
for step in range(max_iter):

    if step % eval_interval == 0:
        out = estimate_loss()
        print(f"step {step} train loss: {out['train']} test loss: {out['test']}")
    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    #print(loss.item())

start = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(start,max_new_tokens=500)[0].tolist()))