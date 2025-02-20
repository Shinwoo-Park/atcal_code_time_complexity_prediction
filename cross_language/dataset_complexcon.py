import pickle

import torch
import torch.utils.data
from torch.utils.data import Dataset

from collate_fns_complexcon import collate_fn, collate_fn_w_aug_complex_con, collate_fn_w_aug_complex_con_double

class dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug

    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["src"] = self.data["tokenized_src"][index]
        else:
            item["src"] = torch.LongTensor(self.data["tokenized_src"][index])

        item["label"] = self.data["label"][index]
        item["idx"] = self.data["idx"][index]

        return item

    def __len__(self):
        return len(self.data["label"])

def get_dataloader(train_batch_size, eval_batch_size, lang, data_type, model, training_type, domain_type, w_aug=True, w_double=False, label_list=None):

    with open("./preprocessed_data/" + lang + "_" + data_type + "_" + model + "_" + training_type + "_" + domain_type + ".pkl", "rb") as f:
        data = pickle.load(f)
    train_dataset = dataset(data["train"],training=True,w_aug=w_aug)
    valid_dataset = dataset(data["valid"],training=False,w_aug=w_aug)
    test_dataset = dataset(data["test"],training=False,w_aug=w_aug)

    if w_double:
        collate_fn_w_aug = collate_fn_w_aug_complex_con_double 
    else:
        collate_fn_w_aug = collate_fn_w_aug_complex_con 

    if w_aug:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn_w_aug, num_workers=0)
    else:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_iter  = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_iter  = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    return train_iter, valid_iter, test_iter