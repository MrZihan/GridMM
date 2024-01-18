import glob
import gzip
import json
import copy

import sys
sys.path.append('/students/u5399302/research/Oscar/Oscar')
from transformers.pytorch_transformers import (BertTokenizer)
tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained(
    '/students/u5399302/research/Oscar/pretrained_models/base-no-labels/ep_67_588997', 
    do_lower_case=True
)
with open('./vocab.txt') as f:
    bert_vocab = [line.rstrip() for line in f]
    

def pad_instr_tokens(instr_tokens, maxlength=20):
    if len(instr_tokens) <= 2: #assert len(raw_instr_tokens) > 2
        return None
    if len(instr_tokens) > maxlength - 2: # -2 for [CLS] and [SEP]
        instr_tokens = instr_tokens[:(maxlength-2)]
    instr_tokens = ['[CLS]'] + instr_tokens + ['[SEP]']
    instr_tokens += ['[PAD]'] * (maxlength-len(instr_tokens))
    assert len(instr_tokens) == maxlength
    return instr_tokens

## ---------------------------------------------------------------------------

MAX_INPUT = 80

data_dir = glob.glob('./*/*_raw.json.gz')
for in_dir in data_dir:

    with gzip.open(in_dir) as f:
        data = json.load(f)
    new_data = copy.deepcopy(data)

    split = in_dir.split('/')[1]
    print('Working on %s split ...'%(split))

    # revise the vocabulary info
    new_data['instruction_vocab']['word_list'] = bert_vocab
    word2idx_dict = {}
    for i, word in enumerate(bert_vocab):
        word2idx_dict[word] = i
    new_data['instruction_vocab']['word2idx_dict'] = word2idx_dict
    new_data['instruction_vocab']['stoi'] = word2idx_dict
    new_data['instruction_vocab']['itos'] = bert_vocab
    new_data['instruction_vocab']['num_vocab'] = len(bert_vocab)
    new_data['instruction_vocab']['UNK_INDEX'] = 100
    new_data['instruction_vocab']['PAD_INDEX'] = 0
    

    # process the instruction for each sample in the data split
    for i, sample in enumerate(new_data['episodes']):
        instr_text = sample['instruction']['instruction_text']

        ''' BERT tokenizer '''
        instr_tokens = tokenizer.tokenize(instr_text)
        padded_instr_tokens = pad_instr_tokens(instr_tokens, MAX_INPUT)
        instr_idxes = tokenizer.convert_tokens_to_ids(padded_instr_tokens)

        new_data['episodes'][i]['instruction']['instruction_tokens'] = instr_idxes
        
        
    json_str = json.dumps(new_data)
    json_bytes = json_str.encode('utf-8')
    with gzip.open('./%s/%s_bertidx.json.gz'%(split,split), 'w') as fout:
        fout.write(json_bytes)

print('Done.')





