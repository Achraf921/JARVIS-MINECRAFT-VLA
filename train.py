import torchvision, json, torch, os, time, random, math, glob, re, gc
import pandas as pd
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from fnmatch import fnmatch
from functools import partial
from concurrent_dl import build_targets, plumbing_ds, collate, worker_init
from utils import wait_for_files, find_assistant_starts, cosine_lr, test_loss, general_part_two_json_parser, build_mixed_batch_tensors
# we just first make sure that everything has been integrally downloaded to disk 

os.environ["TOKENIZERS_PARALLELISM"] = "false" # somehow avoids adding more threads as we will use multiple dataloading cpu processes in
# phase 3

# as I won't be watching over all 30 GPU hours, I first just want to try all loops with limited iterations :

SMOKE = os.environ.get("SMOKE") == "1"

wait_for_files("CraftJarvis/minecraft-vlp", "./data_phase_one.nosync", ["mc-qa-250312.jsonl"])
wait_for_files("CraftJarvis/minecraft-vlp", "./data_phase_two.nosync",
               ["mc-caption-*.jsonl", "mc-grounding-*.jsonl", "mc-vqa-*.jsonl", "images/*.zip"])
wait_for_files("CraftJarvis/minecraft-vla-sft", "./data_phase_three.nosync", ["data/train-*.parquet"])
wait_for_files("CraftJarvis/minecraft-vla-sft", "./data_phase_three_test.nosync", ["data/valid-*.parquet"])
# making sure that all files are integrally downloaded to disk

device = 'cuda'
torch.manual_seed(67) 
random.seed(67)

base_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float32, device_map=device)
base_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct") 
tokenizer = base_processor.tokenizer
with open("./special_tokens.json") as f: # adding our action-specific tokens
    special_tokens = json.load(f)
base_processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

#! PHASE 1):

# We first format the data
path_step_one = "./data_phase_one.nosync/mc-qa-250312.jsonl"

with open(path_step_one) as f:
    lines = f.readlines()

texts = [] # ChatML formatted jsons

for line in lines:
    dct = json.loads(line)
    sys = dct["conversations"][0]["content"][0]['text']
    user = dct["conversations"][1]["content"][0]['text']
    assistant = dct["conversations"][2]["content"][0]['text']
    text = '<|im_start|>system\n'+sys+'<|im_end|>\n'+'<|im_start|>user\n'+user+'<|im_end|>\n'+'<|im_start|>assistant\n'+assistant+'<|im_end|>\n'
    texts.append(text)

random.shuffle(texts)
breaker = int(len(texts) * 0.99)
breaker = int(len(texts)*0.99) # we only keep 1 percent for test set this time
train_ds, test_ds = texts[:breaker], texts[breaker:]

# For phase one we turn all ViT parameter's requires_grad off
for p in base_model.model.visual.parameters():
    p.requires_grad = False

trainable_params = [p for p in base_model.parameters() if p.requires_grad == True]

optimizer = torch.optim.AdamW( # AdamW as in the paper
    params = trainable_params,
    betas=(0.9, 0.95),
    eps=1e-8,
    lr=5e-6,
    weight_decay=0
)

# Phase 1 specific, we will overwrite most of those in the resulting phases
B = 16 # we want batches of 256, we will do gradient accumulation
accum = 256 // B 
max_step = (len(train_ds) // 256) * accum
if SMOKE: max_step = 2 * accum
get_lr = partial(cosine_lr, max_step=max_step, accum=accum, warmup=200)
run_id = time.strftime("%Y%m%d-%H%M%S")
log_dir = f"phase_one_logs.nosync/{run_id}"
os.makedirs(log_dir, exist_ok=True)
train_log = open(f"{log_dir}/train.csv", "w"); train_log.write("step,loss,lr\n")
test_log  = open(f"{log_dir}/test.csv", "w");  test_log.write("step,loss\n")
eval_every = 150

# Phase 1 training loop
base_model.train()
for step in range(max_step):
    batch_tok = base_processor(text=train_ds[step*B:(step+1)*B], padding=True, return_tensors="pt")
    target = build_targets(batch_tok["input_ids"])
    batch_tok, target = batch_tok.to(device), target.to(device)
    if step % accum == 0:
        optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = base_model(**batch_tok).logits
    loss = F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.size(-1)),
                           target[:, 1:].reshape(-1), ignore_index=-67)
    (loss / accum).backward()

    if (step + 1) % accum == 0:
        opt_step = step // accum
        lr = get_lr(opt_step * accum)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        for g in optimizer.param_groups: g["lr"] = lr
        optimizer.step()

        train_log.write(f"{opt_step},{loss.item()},{lr}\n"); train_log.flush()
        if opt_step % 10 == 0:
            print(f"[p1] step {opt_step}/{max_step // accum} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)

        last = step == max_step - 1
        if opt_step % eval_every == 0 or last:
            tl = test_loss(base_model, ((b, build_targets(b["input_ids"])) for b in
                           (base_processor(text=test_ds[i:i+B], padding=True, return_tensors="pt")
                            for i in range(0, len(test_ds), B))), device)
            test_log.write(f"{opt_step},{tl}\n"); test_log.flush()
            print(f"[p1]   test loss {tl:.4f}", flush=True)

path = f"checkpoints/{run_id}/phase1"
base_model.save_pretrained(path); base_processor.save_pretrained(path)
train_log.close(); test_log.close()
# cleaning behind us 
del optimizer, trainable_params
del logits, loss, batch_tok, target 
del lines, texts, train_ds, test_ds
base_model.zero_grad(set_to_none=True)
gc.collect(); torch.cuda.empty_cache() # garbage collection

#! PHASE 2):

root = './data_phase_two.nosync/'
text, path =  general_part_two_json_parser(root+'mc-caption-241104.jsonl')
text2, path2 = general_part_two_json_parser(root+'mc-grounding-point-embodied-image5.jsonl')
text3, path3 = general_part_two_json_parser(root+'mc-grounding-point-embodied.jsonl')
text4, path4 = general_part_two_json_parser(root+'mc-grounding-point-gui.jsonl')
text5, path5 = general_part_two_json_parser(root+'mc-vqa-241102.jsonl')
text = text + text2 + text3 + text4 + text5
path = path + path2 + path3 + path4 + path5  # adding all 5 files together
# let's put them into tuples to shuffle them while keeping order
tups = [(t,p) for t,p in zip(text,path)]
random.shuffle(tups)
# now that it is truly shuffled let's split them back
shuf_text, shuf_path = [], []
for tup in tups:
    shuf_text.append(tup[0])
    shuf_path.append(tup[1])

# let's now split our sets (1% test)
split = len(shuf_text) - int(len(shuf_text)*0.01) # 99% mark
# train
txt_train = shuf_text[:split]
path_train = shuf_path[:split]
# test
txt_test = shuf_text[split:]
path_test = shuf_path[split:]


# similarly to phase 1, btu this time with the ViT params on
for p in base_model.model.visual.parameters():
    p.requires_grad = True
trainable_params = [p for p in base_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, betas=(0.9, 0.95), eps=1e-8, lr=5e-6, weight_decay=0)
B = 8 # this time we will load images in memory so we do need smaller batches                                    
accum = 256 // B
max_step = (len(txt_train) // 256) * accum
if SMOKE: max_step = 2 * accum
get_lr = partial(cosine_lr, max_step=max_step, accum=accum, warmup=200)
log_dir = f"phase_two_logs.nosync/{run_id}"
os.makedirs(log_dir, exist_ok=True)
train_log = open(f"{log_dir}/train.csv", "w"); train_log.write("step,loss,lr\n")
test_log  = open(f"{log_dir}/test.csv", "w");  test_log.write("step,loss\n")

# Phase 2 training loopi
base_model.train()
for step in range(max_step):
    batch_tok = build_mixed_batch_tensors(txt_train[step*B:(step+1)*B], path_train[step*B:(step+1)*B], base_processor)
    target = build_targets(batch_tok["input_ids"])
    batch_tok, target = batch_tok.to(device), target.to(device)
    if step % accum == 0:
        optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = base_model(**batch_tok).logits
    loss = F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.size(-1)),
                           target[:, 1:].reshape(-1), ignore_index=-67)
    (loss / accum).backward()

    if (step + 1) % accum == 0:
        opt_step = step // accum
        lr = get_lr(opt_step * accum)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        for g in optimizer.param_groups: g["lr"] = lr
        optimizer.step()

        train_log.write(f"{opt_step},{loss.item()},{lr}\n"); train_log.flush()
        if opt_step % 10 == 0:
            print(f"[p2] step {opt_step}/{max_step // accum} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)

        last = step == max_step - 1
        if opt_step % eval_every == 0 or last:
            tl = test_loss(base_model, ((b, build_targets(b["input_ids"])) for b in
                           (build_mixed_batch_tensors(txt_test[i:i+B], path_test[i:i+B], base_processor)
                            for i in range(0, len(txt_test), B))), device)
            test_log.write(f"{opt_step},{tl}\n"); test_log.flush()
            print(f"[p2]   test loss {tl:.4f}", flush=True)

path = f"checkpoints/{run_id}/phase2"
base_model.save_pretrained(path); base_processor.save_pretrained(path)
train_log.close(); test_log.close()
# we then garbage all of our variables before phase 3 
del optimizer, trainable_params
del logits, loss, batch_tok, target
del txt_train, path_train, txt_test, path_test, shuf_text, shuf_path, tups
del text, text2, text3, text4, text5, path2, path3, path4, path5
base_model.zero_grad(set_to_none=True)
gc.collect()
torch.cuda.empty_cache()

#! PHASE 3 :

cols = ["conversations", "image_bytes", "label"]
train_table = pq.read_table(sorted(glob.glob("./data_phase_three.nosync/data/train-*.parquet")), columns=cols)
test_table  = pq.read_table(sorted(glob.glob("./data_phase_three_test.nosync/data/valid-*.parquet")), columns=cols)
train_ds, test_ds = plumbing_ds(train_table), plumbing_ds(test_table)

# Vit frozen again in stage 3
for p in base_model.model.visual.parameters():
    p.requires_grad = False
trainable_params = [p for p in base_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, betas=(0.9, 0.95), eps=1e-8, lr=5e-6, weight_decay=0)

B = 8 # since I won't be watching, I am lowkey scared to OOM and loose hours of MONEY to lambda, I'd rather do 8 than try 16
accum = 256 // B
max_step = (len(train_ds) // 256) * accum # we'll try to go for one full epoch though this is gonna burn my pockets innit
if SMOKE: max_step = 2 * accum
get_lr = partial(cosine_lr, max_step=max_step, accum=accum, warmup=200)
save_every = 500

loader = DataLoader(train_ds, batch_size=B, shuffle=True,
                    generator=torch.Generator().manual_seed(67),
                    num_workers=16, persistent_workers=True, worker_init_fn=worker_init,
                    collate_fn=partial(collate, processor=base_processor))

log_dir = f"phase_three_logs.nosync/{run_id}"
os.makedirs(log_dir, exist_ok=True)
train_log = open(f"{log_dir}/train.csv", "w"); train_log.write("step,loss,lr\n")
test_log  = open(f"{log_dir}/test.csv", "w");  test_log.write("step,loss\n")
it = iter(loader)
# Phase 3 training loop
base_model.train()
for step in range(max_step):
    try:
        batch_tok, target, tags = next(it)
    except StopIteration:
        it = iter(loader)
        batch_tok, target, tags = next(it)
    batch_tok, target = batch_tok.to(device), target.to(device)
    if step % accum == 0:
        optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = base_model(**batch_tok).logits
    loss = F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.size(-1)),
                           target[:, 1:].reshape(-1), ignore_index=-67)
    (loss / accum).backward()

    if (step + 1) % accum == 0:
        opt_step = step // accum
        lr = get_lr(opt_step * accum)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        for g in optimizer.param_groups: g["lr"] = lr
        optimizer.step()

        train_log.write(f"{opt_step},{loss.item()},{lr}\n"); train_log.flush()
        if opt_step % 10 == 0:
            print(f"[p3] step {opt_step}/{max_step // accum} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)

        last = step == max_step - 1
        if opt_step % eval_every == 0 or last:
            tl = test_loss(base_model, (collate([test_ds[j] for j in range(i, min(i+B, len(test_ds)))], base_processor)[:2]
                                        for i in range(0, len(test_ds), B)), device)
            test_log.write(f"{opt_step},{tl}\n"); test_log.flush()
            print(f"[p3]   test loss {tl:.4f}", flush=True)

        if opt_step > 0 and opt_step % save_every == 0:
            ckpt = f"checkpoints/{run_id}/phase3/step_{opt_step}"
            base_model.save_pretrained(ckpt); base_processor.save_pretrained(ckpt)

path = f"checkpoints/{run_id}/phase3/final"
base_model.save_pretrained(path); base_processor.save_pretrained(path)
train_log.close(); test_log.close()
# done