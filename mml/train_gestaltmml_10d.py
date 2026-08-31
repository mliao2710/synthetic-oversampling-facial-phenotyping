#!/usr/bin/env python3
from __future__ import annotations

import os

import argparse, ast, csv, json, random, re, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import ViltForQuestionAnswering, ViltProcessor

DB_ROOT = Path(
    os.environ.get("GMDB_DATASET_ROOT", "data/GestaltMatcherDB")
).resolve()

REAL_ROOT = Path(
    os.environ.get("GMDB_IMAGE_ROOT", "data/gmdb_crops")
).resolve()


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Dataset folder name or absolute path')
    p.add_argument('--db-root', type=Path, default=DB_ROOT)
    p.add_argument('--real-image-root', type=Path, default=REAL_ROOT)
    p.add_argument('--synthetic-image-root', type=Path)
    p.add_argument('--synthetic-text', choices=['star','disease_profile'], default='star')
    p.add_argument('--real-augmentation', choices=['matched','matched_star','original3'], default='matched_star')
    p.add_argument('--model-name', default='dandelin/vilt-b32-mlm')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--max-length', type=int, default=40)
    p.add_argument('--output-root', type=Path, default=Path('./runs_10d'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def norm_id(x):
    s = str(x).strip()
    return s[:-2] if re.fullmatch(r'\d+\.0', s) else s


def parse_terms(x):
    if pd.isna(x): return []
    s = str(x).strip()
    q = [(a or b).strip() for a,b in re.findall(r"'([^']+)'|\"([^\"]+)\"", s) if (a or b).strip()]
    if q: return q
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)): return [str(z).strip() for z in v if str(z).strip()]
    except Exception: pass
    return [z.strip(" []'\"") for z in re.split(r'[;,]', s) if z.strip(" []'\"")]


def clean(s):
    return re.sub(r'\s+', ' ', re.sub(r'[(),.]', ' ', s)).strip()


def real_text(row):
    g = 'unknown' if pd.isna(row.get('gender')) else str(row.get('gender'))
    y = 'unknown' if pd.isna(row.get('age_year')) else str(int(row.get('age_year')))
    m = 'unknown' if pd.isna(row.get('age_month')) else str(int(row.get('age_month')))
    e = row.get('ethnicity_sub_category')
    if pd.isna(e) or not str(e).strip(): e = row.get('ethnicity_category')
    e = 'unknown' if pd.isna(e) else str(e)
    terms = parse_terms(row.get('hpo_terms'))
    return clean(f'Sex {g} Age {y} years {m} months Ethnicity {e} ' + ' '.join(terms))


def one_csv(d, pat):
    xs = list(d.glob(pat))
    if len(xs) != 1: raise RuntimeError(f'Expected one {pat} in {d}, found {xs}')
    return xs[0]


def load_tables(ds):
    md = ds/'gmdb_metadata'
    meta = pd.read_csv(md/'gmdb_metadata.csv')
    tr = pd.read_csv(one_csv(md, 'gmdb_train_images_*.csv'))
    va = pd.read_csv(one_csv(md, 'gmdb_val_images_*.csv'))
    for x in (meta,tr,va): x['image_id'] = x['image_id'].map(norm_id)
    tr['label'] = tr['label'].astype(str); va['label'] = va['label'].astype(str)
    return meta,tr,va


def real_img(i, root):
    for n in [f'{i}_aligned.jpg',f'{i}.jpg',f'{i}_aligned.png',f'{i}.png']:
        p=root/n
        if p.is_file(): return p
    raise FileNotFoundError(f'Real image {i} not found under {root}')


def synth_index(roots):
    out={}
    for root in roots:
        if not root or not root.exists(): continue
        print('Indexing synthetic images under', root)
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                out.setdefault(p.stem, p)
                if p.stem.endswith('_aligned'): out.setdefault(p.stem[:-8], p)
    return out


def profiles(real_train):
    d=defaultdict(Counter)
    for _,r in real_train.iterrows(): d[str(r['label'])].update(parse_terms(r.get('hpo_terms')))
    return {k: clean(' '.join(t for t,_ in c.most_common(20))) or '*' for k,c in d.items()}


def build_examples(ds, meta, tr, va, real_root, synth_root, synth_text, real_aug):
    mu = meta.drop_duplicates('image_id')
    tm = tr.merge(mu,on='image_id',how='left',indicator=True,suffixes=('','_meta'))
    vm = va.merge(mu,on='image_id',how='left',indicator=True,suffixes=('','_meta'))
    tm['synthetic']=tm['_merge'].eq('left_only'); vm['synthetic']=vm['_merge'].eq('left_only')
    if vm['synthetic'].any(): raise RuntimeError('Validation contains synthetic/unmatched image IDs')
    labels=sorted(tr.label.unique())
    if len(labels)!=10: raise RuntimeError(f'Expected 10 labels, found {len(labels)}')
    label2id={x:i for i,x in enumerate(labels)}
    prof=profiles(tm[~tm.synthetic])
    roots=[]
    if synth_root: roots.append(synth_root)
    roots += [ds, ds/'images', ds/'pdidb_images', DB_ROOT/'_pdidb_shared_v1.1.0_10d']
    si=synth_index(roots) if tm.synthetic.any() else {}
    train=[]; first=None
    for _,r in tm.iterrows():
        iid,label=str(r.image_id),str(r.label)
        if r.synthetic:
            p=si.get(iid) or si.get(iid+'_aligned')
            if p is None: raise FileNotFoundError(f'Synthetic image {iid} not found; pass --synthetic-image-root')
            train.append(dict(image_id=iid,image_path=str(p),text='*' if synth_text=='star' else prof.get(label,'*'),label=label,kind='synthetic'))
        else:
            p=real_img(iid,real_root); first=first or p; txt=real_text(r)
            train.append(dict(image_id=iid,image_path=str(p),text=txt,label=label,kind='real_matched'))
            if real_aug in {'matched_star','original3'}:
                train.append(dict(image_id=iid,image_path=str(p),text='*',label=label,kind='real_star'))
            if real_aug=='original3':
                train.append(dict(image_id=iid,image_path=str(first),text=txt,label=label,kind='original_fixed_image'))
    val=[dict(image_id=str(r.image_id),image_path=str(real_img(str(r.image_id),real_root)),text=real_text(r),label=str(r.label),kind='validation') for _,r in vm.iterrows()]
    return train,val,label2id


class DS(Dataset):
    def __init__(self, ex, proc, l2i, maxlen): self.ex,self.proc,self.l2i,self.maxlen=ex,proc,l2i,maxlen
    def __len__(self): return len(self.ex)
    def __getitem__(self,i):
        e=self.ex[i]
        with Image.open(e['image_path']) as im:
            z=self.proc(images=im.convert('RGB'),text=e['text'],padding='max_length',truncation=True,max_length=self.maxlen,return_tensors='pt')
        z={k:v.squeeze(0) for k,v in z.items()}
        y=torch.zeros(len(self.l2i)); y[self.l2i[e['label']]]=1.; z['labels']=y
        return z


def collate(proc):
    def f(b):
        pix=[x['pixel_values'] for x in b]; pad=proc.image_processor.pad(pix,return_tensors='pt')
        return {'input_ids':torch.stack([x['input_ids'] for x in b]),'attention_mask':torch.stack([x['attention_mask'] for x in b]),'token_type_ids':torch.stack([x['token_type_ids'] for x in b]),'pixel_values':pad['pixel_values'],'pixel_mask':pad['pixel_mask'],'labels':torch.stack([x['labels'] for x in b])}
    return f


@torch.inference_mode()
def evaluate(model, loader, dev):
    model.eval(); n=t1=t3=t5=0; cc=Counter(); ct=Counter()
    for b in tqdm(loader,desc='Validation',leave=False):
        y=b.pop('labels').argmax(1).to(dev); b={k:v.to(dev,non_blocking=True) for k,v in b.items()}
        q=model(**b).logits.topk(min(5,len(model.module.config.id2label) if hasattr(model,'module') else len(model.config.id2label)),1).indices
        n+=len(y); t1+=q[:,:1].eq(y[:,None]).any(1).sum().item(); t3+=q[:,:3].eq(y[:,None]).any(1).sum().item(); t5+=q.eq(y[:,None]).any(1).sum().item()
        for a,z in zip(y.tolist(),q[:,0].tolist()): ct[a]+=1; cc[a]+=int(a==z)
    return dict(top1=t1/n,top3=t3/n,top5=t5/n,mean_top1=float(np.mean([cc[k]/ct[k] for k in ct])),n=n)


def main():
    a=args(); seed_all(a.seed)
    ds=Path(a.dataset) if Path(a.dataset).is_absolute() else a.db_root/a.dataset
    if not ds.is_dir(): raise FileNotFoundError(ds)
    run=a.output_root/f'{ds.name}_seed{a.seed}_{time.strftime("%Y%m%d_%H%M%S")}'; run.mkdir(parents=True)
    meta,tr,va=load_tables(ds)
    train,val,l2i=build_examples(ds,meta,tr,va,a.real_image_root,a.synthetic_image_root,a.synthetic_text,a.real_augmentation)
    summary={'dataset':str(ds),'raw_train':len(tr),'raw_val':len(va),'train_examples':len(train),'val_examples':len(val),'kinds':dict(Counter(x['kind'] for x in train)),'labels':l2i,'synthetic_text':a.synthetic_text,'real_augmentation':a.real_augmentation,'seed':a.seed}
    (run/'dataset_summary.json').write_text(json.dumps(summary,indent=2)); pd.DataFrame(train).to_csv(run/'train_examples.csv',index=False); pd.DataFrame(val).to_csv(run/'validation_examples.csv',index=False)
    print(json.dumps(summary,indent=2))
    if a.dry_run: print('Dry run complete:',run); return
    if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU detected; submit to a GPU node')
    dev=torch.device('cuda'); proc=ViltProcessor.from_pretrained(a.model_name)
    tl=DataLoader(DS(train,proc,l2i,a.max_length),batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=True,collate_fn=collate(proc))
    vl=DataLoader(DS(val,proc,l2i,a.max_length),batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True,collate_fn=collate(proc))
    i2l={i:x for x,i in l2i.items()}; model=ViltForQuestionAnswering.from_pretrained(a.model_name,num_labels=10,id2label=i2l,label2id=l2i,ignore_mismatched_sizes=True)
    if torch.cuda.device_count()>1: model=torch.nn.DataParallel(model)
    model.to(dev); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate); amp=not a.no_amp; scaler=torch.amp.GradScaler('cuda',enabled=amp)
    fields=['epoch','train_loss','top1','top3','top5','mean_top1','n'];
    with open(run/'metrics.csv','w',newline='') as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    best=(-1,None)
    for ep in range(1,a.epochs+1):
        model.train(); loss_sum=seen=0
        for b in tqdm(tl,desc=f'Epoch {ep}/{a.epochs}'):
            b={k:v.to(dev,non_blocking=True) for k,v in b.items()}; opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',dtype=torch.float16,enabled=amp):
                loss=model(**b).loss
                if loss.ndim: loss=loss.mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); bs=len(b['labels']); loss_sum+=loss.item()*bs; seen+=bs
        m=evaluate(model,vl,dev); row={'epoch':ep,'train_loss':loss_sum/seen,**m}
        print(f"Epoch {ep}: loss={row['train_loss']:.4f} Top1={m['top1']:.4%} MeanTop1={m['mean_top1']:.4%} Top3={m['top3']:.4%} Top5={m['top5']:.4%}")
        with open(run/'metrics.csv','a',newline='') as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
        raw=model.module if hasattr(model,'module') else model
        torch.save({'epoch':ep,'model_state_dict':raw.state_dict(),'metrics':row,'label2id':l2i},run/'last.pt')
        if m['top1']>best[0]: best=(m['top1'],ep); torch.save({'epoch':ep,'model_state_dict':raw.state_dict(),'metrics':row,'label2id':l2i},run/'best.pt'); proc.save_pretrained(run/'processor')
    print('Done. Best epoch:',best[1],'Top1:',best[0]); print('Results:',run)

if __name__=='__main__': main()
