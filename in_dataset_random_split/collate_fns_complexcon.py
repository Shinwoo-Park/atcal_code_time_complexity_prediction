import torch
import itertools

def collate_fn(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 512 # Maximum length of a source code snippet

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            end = min(lengths[i], N)
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    src_batch, src_attn_mask, src_lengths = merge(item_info['src'])

    d={}
    d["label"] = item_info["label"]
    d["src"] = src_batch.cuda()
    d["src_attn_mask"] = src_attn_mask.cuda()
    d["idx"] = item_info["idx"]

    return d

def collate_fn_w_aug_complex_con(data): 

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 512 # Maximum length of a source code snippet

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            end = min(lengths[i], N)

            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask, lengths

    item_info = {}

    for key in data[0].keys():

        item_info[key] = [d[key] for d in data]
            
        flat = itertools.chain.from_iterable(item_info[key])
        
        original_srcs = []
        probs_or_references = []
        
        for i, item in enumerate(flat):
            if i % 2 == 0:
                original_srcs.append(item)
            else:
                probs_or_references.append(item)
                
        srcs_n_probs_or_references = original_srcs + probs_or_references

        item_info[key] = srcs_n_probs_or_references

    ## input
    src_batch, src_attn_mask, src_lengths = merge(item_info['src'])

    d={}
    d["label"] = item_info["label"]
    d["src"] = src_batch.cuda()
    d["src_attn_mask"] = src_attn_mask.cuda()
    d["idx"] = item_info["idx"]

    return d

def collate_fn_w_aug_complex_con_double(data): 

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 512 # Maximum length of a source code snippet

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            end = min(lengths[i], N)

            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask, lengths

    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

        flat = itertools.chain.from_iterable(item_info[key])
        
        original_srcs = []
        problem_descriptions = []
        reference_codes = []
        for i, item in enumerate(flat):

            if i % 3 == 0:
                original_srcs.append(item)
            elif i % 3 == 1:
                problem_descriptions.append(item)
            else:
                reference_codes.append(item)

        srcs_n_probs_and_references = original_srcs + problem_descriptions + reference_codes

        item_info[key] = srcs_n_probs_and_references

    ## input
    src_batch, src_attn_mask, src_lengths = merge(item_info['src'])

    d={}
    d["label"] = item_info["label"]
    d["src"] = src_batch.cuda()
    d["src_attn_mask"] = src_attn_mask.cuda()
    d["idx"] = item_info["idx"]

    return d