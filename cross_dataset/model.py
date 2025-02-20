from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

class primary_encoder_v2_no_pooler_for_con(nn.Module):

    def __init__(self,hidden_size,label_size,model="CodeBERT"):
        super(primary_encoder_v2_no_pooler_for_con, self).__init__()

        self.hidden_size = hidden_size
        self.label_size = label_size
        self.model = model

        if model == "CodeBERT":
            options_name = "microsoft/codebert-base"
            self.encoder_supcon = AutoModel.from_pretrained(options_name,num_labels=label_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=False
        elif model == "GraphCodeBERT":
            options_name = "microsoft/graphcodebert-base"
            self.encoder_supcon = AutoModel.from_pretrained(options_name,num_labels=label_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=False
        elif model == "UniXcoder":
            options_name = "microsoft/unixcoder-base"
            self.encoder_supcon = AutoModel.from_pretrained(options_name,num_labels=label_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=False
        elif model == 'CodeT5+':
            options_name = "Salesforce/codet5p-220m"
            self.encoder_supcon = AutoModel.from_pretrained(options_name,num_labels=label_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=False         
        else:
            raise NotImplementedError

        self.pooler_dropout = nn.Dropout(0.1)
        self.label = nn.Linear(hidden_size, label_size)

    def get_cls_features_ptrnsp(self, text, attn_mask):
        
        if self.model == 'CodeT5+':
            supcon_fea = self.encoder_supcon(input_ids=text, attention_mask=attn_mask, decoder_input_ids=text, decoder_attention_mask=attn_mask, output_hidden_states=True, output_attentions=True, return_dict=True)
            norm_supcon_fea_cls = F.normalize(supcon_fea['decoder_hidden_states'][-1][:,0,:], dim=1)  
            pooled_supcon_fea_cls = supcon_fea['decoder_hidden_states'][-1][:,0,:]
        else:
            supcon_fea = self.encoder_supcon(text,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)
            norm_supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:,0,:], dim=1) # normalized last layer's first token ([CLS])
            pooled_supcon_fea_cls = supcon_fea.pooler_output # [huggingface] Last layer hidden-state of the first token of the sequence (classification token) **further processed by a Linear layer and a Tanh activation function.** The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        return pooled_supcon_fea_cls, norm_supcon_fea_cls

    def forward(self, pooled_supcon_fea_cls):
        supcon_fea_cls_logits = self.label(self.pooler_dropout(pooled_supcon_fea_cls))

        return supcon_fea_cls_logits