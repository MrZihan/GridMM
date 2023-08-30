import os
import json
import numpy as np

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    dataset = 'rxr'
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset, split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc.jsonl' % (dataset, split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)
           
            with open(filepath, "r") as f:
                new_data = []
                 
                for line in f.readlines():                          #依次读取每行  
                    line = line.strip()                             #去掉每行头尾空白  
                    item = json.loads(line)
                    new_data.append(item)

            if split == 'val_train_seen':
                new_data = new_data[:50]

        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split, "r") as f:
                new_data = []
                 
                for line in f.readlines():                          #依次读取每行  
                    line = line.strip()                             #去掉每行头尾空白  
                    item = json.loads(line)
                    new_data.append(item)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
 
        data.append(item)
    return data