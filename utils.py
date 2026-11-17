import os, time, math, torch, json
from fnmatch import fnmatch
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
import torch.nn.functional as F
from PIL import Image

api = HfApi()
# helpers to make sure all data is downloaded and integral on disk
def remote_sizes(repo, patterns):
    return {f.path: f.size
            for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
            if isinstance(f, RepoFile) and any(fnmatch(f.path, p) for p in patterns)}

def wait_for_files(repo, local_dir, patterns):
    want = remote_sizes(repo, patterns)       
    assert want, f"no remote files match {patterns}"
    while True:
        bad = [p for p, s in want.items()
               if not os.path.exists(f"{local_dir}/{p}") or os.path.getsize(f"{local_dir}/{p}") != s]
        if not bad: return
        print(f"{local_dir}: {len(bad)}/{len(want)} files not ready"); time.sleep(60)


# in phase 1, to find the starting token of the assistant section

def find_assistant_starts(token_buckets, buckets):
    # magic tokens: [151644, 77091, 198] = <|im_start|>assistant\n
    dsx_idx = []
    for datapoint, n in zip(token_buckets, range(len(token_buckets))):
        # we'll use sliding window on each sequence to detect starting index
        datapoint = datapoint['input_ids']
        x = -1
        length = len(datapoint)
        for i in range(length - 2):
            if datapoint[i] == 151644 and datapoint[i+1] == 77091:  # our magic tokens
                if datapoint[i+2] == 198:  # \n didn't get merged into a bigger token, we skip it
                    x = i + 3  # x stores ds[:x] for x in range(x, len-1), y = ds[x] is the next token to predict
                else:
                    x = i + 2
        if x == -1:
            print(x)
            print(n)  # to know which line is dirty
            print(buckets[n])
            print(token_buckets[n])
            raise RuntimeError("we have empty x's and y's")
        else:
            dsx_idx.append(x)
    return dsx_idx

# lr function (cosine learning ratw w/)
def cosine_lr(step, max_step, accum=1, lr_max=5e-6, warmup=200):
    # step and max_step in micro-steps; warmup in optimizer steps
    warmup = warmup * accum
    if step <= warmup:
        return step * lr_max / warmup
    progress = (step - warmup) / (max_step - warmup)
    return 0.5 * lr_max * (1 + math.cos(progress * math.pi))

@torch.no_grad() # shared loss generating function
def test_loss(model, batches, device):
    # batches: iterable of (batch_dict, target), target unshifted with -67 masking
    model.eval()
    total, n_tok = 0.0, 0
    for batch, target in batches:
        batch, target = batch.to(device), target.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits
        total += F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.size(-1)),
                                 target[:, 1:].reshape(-1),
                                 ignore_index=-67, reduction="sum").item()
        n_tok += (target[:, 1:] != -67).sum().item()
    model.train()
    return total / n_tok

def general_part_two_json_parser(json_filepath, root='./data_phase_two.nosync'): # our phase 2 general json parser
    texts, paths = [], []
    with open(json_filepath) as f:
        for l in f:
            line = json.loads(l)
            images = [f"{root}/{p}" for p in line['image']]
            k, s = 0, ""
            for turn in line['conversations']:
                content = turn['content'] if isinstance(turn['content'], list) else [{"type": "text", "text": turn['content']}]
                body, points = "", []
                for obj in content:
                    if obj['type'] == 'text':
                        body += obj['text']
                    elif obj['type'] == 'image':
                        body += "<|vision_start|><|image_pad|><|vision_end|>" # path goes in images
                        k += 1
                    elif obj['type'] == 'point':# 0-100 -> Qwen3-VL 0-1000
                        for p in obj['point']:
                            x, y = (p['x'], p['y']) if isinstance(p, dict) else p
                            points.append({"point_2d": [min(max(round(x*10), 0), 1000), min(max(round(y*10), 0), 1000)],
                                           "label": obj.get('label', '')})
                    else:
                        raise ValueError(f"unknown content type {obj['type']!r} in row {line.get('id')}")
                if points:
                    body += json.dumps(points)
                s += f"<|im_start|>{turn['role']}\n{body}<|im_end|>\n"
            if k != len(images):
                raise ValueError(f"{k} image slots but {len(images)} paths in row {line.get('id')}")
            texts.append(s)
            paths.append(images)
    return texts, paths

def build_mixed_batch_tensors(texts, paths, processor): # processor passed in this time utils.py can't see train.py's base_processor
    images = [Image.open(p).convert("RGB") for row in paths for p in row]
    for im in images: im.thumbnail((640, 640))
    return processor(text=texts, images=images or None, padding=True, return_tensors="pt")   # `or None`: safe if a batch has no images