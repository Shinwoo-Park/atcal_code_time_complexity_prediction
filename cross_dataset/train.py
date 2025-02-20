import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
import time
import torch
import torch.utils.data
from torch import nn
from util import iter_product
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup
import train_config as train_config
from dataset_complexcon import get_dataloader
import loss_complexcon as loss
from model import primary_encoder_v2_no_pooler_for_con
import sys
import pickle 
import logging
import pandas as pd
from typing import List
from torch.nn import Softmax
logger = logging.getLogger(__name__)

def train(epoch,train_loader,model_main,loss_function,optimizer,lr_scheduler,log,save_home):

    model_main.cuda()
    model_main.train()

    total_true,total_pred_1,acc_curve_1 = [],[],[]
    train_loss_1 = 0
    total_epoch_acc_1 = 0
    steps = 0
    start_train_time = time.time()
        
    # Adjust train_batch_size 
    if log.param.w_aug:
        if log.param.w_double:
            train_batch_size = log.param.train_batch_size*3
        else:
            train_batch_size = log.param.train_batch_size*2 
    else:
        train_batch_size = log.param.train_batch_size
    
    print("train with aug:", log.param.w_aug)
    print("train with double aug:", log.param.w_double)
    print("train with separate double aug:", log.param.w_separate)
    print("loss with sup(using label info):", log.param.w_sup)
    print("len(train_loader):", len(train_loader))
    print("train_batch_size including problem description/reference code:", train_batch_size)
    
    pred_1 = None 
    only_original_labels = None
    train_ids = None
    supcon_feature_1_list = []
    for idx,batch in enumerate(train_loader):
        text_name = "src"
        label_name = "label"
        idx_name = "idx"
        
        text = batch[text_name]
        attn = batch[text_name+"_attn_mask"]
        label = batch[label_name]
        label = torch.tensor(label)
        label = torch.autograd.Variable(label).long()
        ids = batch[idx_name]   

        if (label.size()[0] is not train_batch_size):
            continue

        if torch.cuda.is_available():
            text = text.cuda()
            attn = attn.cuda()
            label = label.cuda()

        #####################################################################################
        if log.param.w_aug: # @CAL or PCL or RCL
            if log.param.w_double: # @CAL
                if log.param.w_separate: # Compute Contrastive Learning Loss separately for each augmented text (Problem Description, Reference Code)
                    assert log.param.train_batch_size == label.shape[0] // 3 
                    assert label.shape[0] % 3 == 0
                    original_label, augmented_label_1, augmented_label_2 = torch.split(label, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    only_original_labels = original_label

                    original_text, augmented_text_1, augmented_text_2 = torch.split(text, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    original_attn, augmented_attn_1, augmented_attn_2 = torch.split(attn, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)

                    original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn) 

                    _, augmented_supcon_feature_1_1 = model_main.get_cls_features_ptrnsp(augmented_text_1,augmented_attn_1)
                    _, augmented_supcon_feature_1_2 = model_main.get_cls_features_ptrnsp(augmented_text_2,augmented_attn_2)

                    supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1_1], dim=0)
                    supcon_feature_2 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1_2], dim=0)

                    assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size 

                    pred_1 = model_main(original_last_layer_hidden_states)

                else: # Compute Contrastive Learning Loss for both augmented texts (Problem Description, Reference Code) together
                    assert log.param.train_batch_size == label.shape[0] // 3 
                    assert label.shape[0] % 3 == 0
                    original_label, augmented_label_1, augmented_label_2 = torch.split(label, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    only_original_labels = original_label

                    original_text, augmented_text_1, augmented_text_2 = torch.split(text, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    original_attn, augmented_attn_1, augmented_attn_2 = torch.split(attn, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)

                    original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn)

                    _, augmented_supcon_feature_1_1 = model_main.get_cls_features_ptrnsp(augmented_text_1,augmented_attn_1)
                    _, augmented_supcon_feature_1_2 = model_main.get_cls_features_ptrnsp(augmented_text_2,augmented_attn_2)

                    supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1_1, augmented_supcon_feature_1_2], dim=0)
                    
                    assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size 

                    pred_1 = model_main(original_last_layer_hidden_states)

            else: # PCL or RCL
                assert log.param.train_batch_size == label.shape[0] // 2 
                assert label.shape[0] % 2 == 0
                original_label, augmented_label = torch.split(label, [log.param.train_batch_size, log.param.train_batch_size], dim=0)
                only_original_labels = original_label

                original_text, augmented_text = torch.split(text, [log.param.train_batch_size, log.param.train_batch_size], dim=0)
                original_attn, augmented_attn = torch.split(attn, [log.param.train_batch_size, log.param.train_batch_size], dim=0)

                original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn) 
                
                _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(augmented_text,augmented_attn) 

                supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1], dim=0)
                assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size 

                pred_1 = model_main(original_last_layer_hidden_states)

        ### Cross-Entropy Loss
        else: 
            assert log.param.train_batch_size == label.shape[0] 
            only_original_labels = label
            original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn)
            pred_1 = model_main(original_last_layer_hidden_states)

        if log.param.save_td_w_features == True:                   
            if train_ids is None:
                train_ids = list(ids)
            else:
                train_ids.extend(list(ids))
            supcon_feature_1_list.append(original_supcon_feature_1.detach().cpu().numpy())

        ###################################################################################################
        ### Compute loss
        if log.param.w_aug: # @CAL or PCL or RCL
            if log.param.w_double: # @CAL
                if log.param.w_separate: # Compute Contrastive Learning Loss separately for each augmented text (Problem Description, Reference Code)
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((log.param.w1*(1-loss_function["lambda_loss"]))*loss_function["contrastive"](features=supcon_feature_1, labels=None)) + (((1-log.param.w1)*(1-loss_function["lambda_loss"]))*loss_function["contrastive"](features=supcon_feature_2, labels=None)) 
                else: # Compute Contrastive Learning Loss for both augmented texts (Problem Description, Reference Code) together
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive_for_double"](features=supcon_feature_1, labels=None)) 
            else: # PCL or RCL
                loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive"](features=supcon_feature_1, labels=None))
        else: # Cross-Entropy Loss 
            loss_1 = loss_function["ce_loss"](pred_1,only_original_labels)

        loss = loss_1
        train_loss_1  += loss_1.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)
        optimizer.step()
        model_main.zero_grad()
        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print (f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss_1: {loss_1.item():.4f}, Time taken: {((time.time()-start_train_time)/60): .2f} min')
            start_train_time = time.time()

        true_list = only_original_labels.data.detach().cpu().tolist()
        total_true.extend(true_list)

        num_corrects_1 = (torch.max(pred_1, 1)[1].view(only_original_labels.size()).data == only_original_labels.data).float().sum()
        pred_list_1 = torch.max(pred_1, 1)[1].view(only_original_labels.size()).data.detach().cpu().tolist()

        total_pred_1.extend(pred_list_1) 

        acc_1 = 100.0 * (num_corrects_1/log.param.train_batch_size)
        acc_curve_1.append(acc_1.item())
        total_epoch_acc_1 += acc_1.item()

    print(train_loss_1/len(train_loader))
    print(total_epoch_acc_1/len(train_loader))
    
    if log.param.save_td_w_features == True: 
        supcon_feature_1_list = np.concatenate(supcon_feature_1_list, axis=0)

    return train_loss_1/len(train_loader), total_epoch_acc_1/len(train_loader), acc_curve_1, supcon_feature_1_list, train_ids

def test(test_loader,model_main,log):
    model_main.eval()
    
    total_pred_1,total_true,total_pred_prob_1 = [],[],[]
    save_pred = {"true":[],"pred_1":[],"pred_prob_1":[],"feature":[]}

    total_feature = []
    total_num_corrects = 0
    total_num = 0
    print(len(test_loader))
    with torch.no_grad():
        for idx,batch in enumerate(test_loader):
            text_name = "src"
            label_name = "label"

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]
            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                text = text.cuda()
                attn = attn.cuda()
                label = label.cuda()

            last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn) 
            pred_1 = model_main(last_layer_hidden_states)

            num_corrects_1 = (torch.max(pred_1, 1)[1].view(label.size()).data == label.data).float().sum()

            pred_list_1 = torch.max(pred_1, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature_1.data.detach().cpu().tolist())
            total_pred_prob_1.extend(pred_1.data.detach().cpu().tolist())

    f1_score_1 = 100 * f1_score(total_true,total_pred_1, average="macro")
    f1_score_1_w = 100 * f1_score(total_true,total_pred_1, average="weighted")
    f1_score_1 = {"macro":f1_score_1,"weighted":f1_score_1_w}

    total_acc = 100 * total_num_corrects / total_num

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1

    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1

    return total_acc, f1_score_1, save_pred

def cl_train(log):

    np.random.seed(log.param.model_seed)
    random.seed(log.param.model_seed)
    torch.manual_seed(log.param.model_seed)
    torch.cuda.manual_seed(log.param.model_seed)
    torch.cuda.manual_seed_all(log.param.model_seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    if log.param.w_separate:
        log.param.w_double = True
    origin_training_type = None
    if log.param.training_type == "ProbCon" or log.param.training_type == "RefCon" or log.param.training_type == "Prob_RefCon":
        log.param.w_sup = False
        log.param.w_aug = True
    if log.param.training_type == "Prob_RefCon":
        log.param.w_double = True
        log.param.w_aug = True

    print("#######################start run#######################")
    print("log:", log)

    train_data,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.lang,log.param.data_type,log.param.model,log.param.training_type,log.param.domain_type,w_aug=log.param.w_aug,w_double=log.param.w_double,label_list=None)
    print("len(train_data):", len(train_data)) 

    if origin_training_type is not None:
        log.param.training_type = origin_training_type

    losses = {"contrastive":loss.SupConLoss(temperature=log.param.temperature),"ce_loss":nn.CrossEntropyLoss(),"lambda_loss":log.param.lambda_loss,"contrastive_for_double":loss.SupConLoss_for_double(temperature=log.param.temperature)}

    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model_main = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model)

    total_params = list(model_main.named_parameters())
    num_training_steps = int(len(train_data)*log.param.nepoch)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)], 'weight_decay': log.param.decay},
    {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate, eps=log.param.eps)     
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps//5, num_training_steps=num_training_steps)

    # save directory 
    save_home = "./save/" + log.param.lang + "/" + log.param.data_type + "/" + log.param.model + "/" + log.param.training_type + "/" + str(log.param.model_seed) + "/" + str(log.param.lambda_loss) + "/"
    
    total_train_acc_curve_1, total_val_acc_curve_1 = [],[]

    supcon_feature_1_epoch_dict,train_ids_epoch_dict = {}, {}
    best_epoch = 0
    for epoch in range(1, log.param.nepoch + 1):

        train_loss_1,train_acc_1,train_acc_curve_1,supcon_feature_1_list,train_ids = train(epoch,train_data,model_main,losses,optimizer,lr_scheduler,log,save_home)
        supcon_feature_1_epoch_dict[epoch] = supcon_feature_1_list
        train_ids_epoch_dict[epoch] = train_ids
        val_acc_1,val_f1_1,val_save_pred = test(valid_data,model_main,log)
        test_acc_1,test_f1_1,test_save_pred = test(test_data,model_main,log)

        total_train_acc_curve_1.extend(train_acc_curve_1)

        print('====> Epoch: {} Train loss_1: {:.4f}'.format(epoch, train_loss_1))

        os.makedirs(save_home,exist_ok=True)

        if epoch == 1:
            best_criterion = 0.0

        is_best = val_f1_1["macro"] > best_criterion
        best_criterion = max(val_f1_1["macro"], best_criterion)
        print("Best model evaluated by Macro F1")
        print(f'Valid Accuracy: {val_acc_1:.2f} Valid F1: {val_f1_1["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc_1:.2f} Test F1: {test_f1_1["macro"]:.2f}')

        if is_best:
            print("======> Best epoch <======")
            log.train_loss_1 = train_loss_1
            log.stop_epoch = epoch
            log.valid_f1_score_1 = val_f1_1
            log.test_f1_score_1 = test_f1_1
            log.valid_accuracy_1 = val_acc_1
            log.test_accuracy_1 = test_acc_1
            log.train_accuracy_1 = train_acc_1
            best_epoch = epoch

            ## load the model
            with open(save_home+"/log.json", 'w') as fp:
                json.dump(dict(log), fp,indent=4)
            fp.close()

            # save model
            if log.param.save:
                torch.save(model_main.state_dict(), os.path.join(save_home, 'model.pt'))
                print(f"best model is saved at {os.path.join(save_home, 'model.pt')}")

    # save features 
    if log.param.save_td_w_features == True: 
        with open(save_home + "supcon_feature_1_" + str(best_epoch) + ".pickle", 'wb') as fp:
            pickle.dump(supcon_feature_1_epoch_dict[best_epoch], fp)
        with open(save_home + "train_ids_" + str(best_epoch) + ".pickle", 'wb') as fp:
            pickle.dump(train_ids_epoch_dict[best_epoch], fp)

if __name__ == '__main__':

    tuning_param = train_config.tuning_param

    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list))

    for param_com in param_list[1:]:

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        # constant, linear, logn, quadratic, cubic, nlogn, exponential
        log.param.label_size = 7 
        
        cl_train(log)