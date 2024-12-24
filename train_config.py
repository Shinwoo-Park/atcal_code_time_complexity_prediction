# Experiment Main Settings
model = ["UniXcoder"]
lang = ["java", "python"] 
data_type = ["problem_split"] 
model_seed = [42]
domain_type = ["in_domain"]

# Hyperparameter
lambda_loss = [0.75]
w1 = [0.5]
main_learning_rate = [2e-6] 
train_batch_size = [4] 
eval_batch_size = [4] 
nepoch = [20]  
temperature = [0.3] 
decay = [1e-2] 
eps = [1e-8]

# Experiment Configuration
hidden_size = 768
save = False 
save_td_w_features = False
w_aug = True 
w_sup = False 
w_separate = False  
w_double = True  
training_type = "Prob_RefCon"

tuning_param  = ["model","lang","data_type","model_seed","lambda_loss","main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","decay","eps","domain_type","w1"] 

param = {"temperature":temperature,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"lambda_loss":lambda_loss,"decay":decay,"model_seed":model_seed,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate, "eps":eps, "training_type":training_type, "lang":lang, "data_type":data_type, "model":model, "save_td_w_features":save_td_w_features, "domain_type":domain_type, "w1":w1}