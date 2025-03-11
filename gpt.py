import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW

block_size = 8
batch_size = 32
max_iter = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 500
eval_iter = 200
n_embed = 32
learning_rate = 1e-3


text = ''
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
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
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones((block_size,block_size))))
    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        out = wei @ v
        return out
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
        self.sa_head = Head(n_embed)
    def forward(self, idx,target=None):
        B,T = idx.shape
        tok_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_embed + pos_embed
        x = self.sa_head(x)
        logits = self.lm_head(x)
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
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx_cond)
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



    

model = BigramLanguageModel()
model = model.to(device)
optim = AdamW(model.parameters(),lr=learning_rate)
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