import torch
import pyarrow
import io
from PIL import Image

# we need this to map from arrow objects to python objects in the DataLoader
class plumbing_ds(torch.utils.data.Dataset):
    def __init__(self, table : pyarrow.Table):
        super().__init__()
        self.table = table
    def __len__(self):
        return self.table.num_rows
    def __getitem__(self, i):
        return self.table.slice(i, 1).to_pylist()[0]

def build_targets(ids): # for the masked y 
    target = torch.full_like(ids, -67)
    for r, line in enumerate(ids.tolist()):
        for i in range(len(line) - 2):
            if line[i] == 151644 and line[i+1] == 77091: # every <|im_start|>assistant
                j = i + 3
                end = line.index(151645, j) # its own <|im_end|>
                target[r, j:end+1] = ids[r, j:end+1]
    return target

def collate(rows, processor): # since we have multiple processes, it is not really on the critical path, so no need to worry too much about optimizing
    strings, images, tags = [], [], []
    for row in rows:
        tags.append(row['label'])
        images.append(Image.open(io.BytesIO(row["image_bytes"][0])).convert("RGB"))
        row = row['conversations']
        string = ''
        string += '<|im_start|>user\n' + row[0]['content'][0]['text'] # user prompt
        string += '<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n' # image part, the im_end token of user prompt comes here
        string += '<|im_start|>assistant\n' + row[1]['content'][0]['text'] + '<|im_end|>\n'
        strings.append(string)
    batch = processor(text=strings, images=images, padding=True, return_tensors="pt")
    targets = build_targets(batch['input_ids'])
    return batch, targets, tags


# should process a list of rows, and return a tensor
# format (in python object) is something like this
'''
[
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "Break the acacia leaves to collect them.\nobservation: \n"
      },
      {
        "type": "image",
        "text": "<image>"
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "<|reserved_special_token_178|><|reserved_special_token_204|><|reserved_special_token_216|><|reserved_special_token_230|><|reserved_special_token_179|>\n"
      }
    ]
  }
  ]
'''