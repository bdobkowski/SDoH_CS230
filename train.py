from transformers import AutoTokenizer, AutoModel
import transformers
from util.data_cleaning import load_data, weighted_sampler
from util.dataset import BertDataset
import torch
from model.model import BertPretrained
import mlvt
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import ray
import os
from ray import tune
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = 0
# device = 'cpu'

if len(sys.argv) > 1:
    run_weak_labels = True if sys.argv[1] == "weak" else False
else:
    run_weak_labels = False

NUM_EPOCHS     = 50
weak_label_words = ['bacs','wesley','melanoma','burned', 'prison']

# Config for hyperparameter tuning
config = {
    "optimizer": tune.grid_search([torch.optim.Adam, torch.optim.SGD]),
    "lr": tune.grid_search([1e-4, 1e-3, 1e-2]),
    "prob_threshold": tune.grid_search([0.5, 0.6]),
    "batch_size": tune.grid_search([128, 256]),
    "max_len": tune.grid_search([128, 256])
}

def train_model(config, modelname, labeling):
    # tune.utils.wait_for_gpu()
    batch_size = config["batch_size"]
    max_len = config["max_len"]
    pretrained_model = modelname

    if labeling == "supervised":
        print('Running supervised learning')
        X_train, X_test, y_train, y_test = load_data("/home/ubuntu/SDoH_CS230/data/final_foodinsecurity_data.csv") # weak labeling 
    elif labeling == "weak":
        print('Running weak labeling')
        X_train, X_test, y_train, y_test = load_data("/home/ubuntu/SDoH_CS230/data/final_foodinsecurity_data.csv",
                                                      unstructured_data="/home/ubuntu/SDoH_CS230/data/raw_weak_labels.csv",
                                                      weak_label_words=weak_label_words)
    else:
        raise Exception('No such labeling technique - see main function of train.py')
    # print(f'number training examples: {len(X_train)}')
    # print(f'number of labels: {len(y_train)}')
    # print(f'number positive training labels: {np.sum(y_train)}')
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model,
                                                           do_lower_case=True, truncation=True, 
                                                           padding='True', pad_to_max_length=True,
                                                           add_special_tokens=True)
    train_dataset = BertDataset(text=X_train.values, tokenizer=tokenizer,
                                        max_len=max_len, target=y_train.values)
    test_dataset = BertDataset(text=X_test.values, tokenizer=tokenizer,
                                        max_len=max_len, target=y_test.values)
    sampler = weighted_sampler(torch.from_numpy(y_train.values))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, sampler=sampler,
                                               num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, len(test_dataset),
                                               shuffle=True, num_workers=1)
    
    model = BertPretrained(pretrained_model)
    model = model.to(device) 
    
    # Freeze pretrained model params
    for param in model.parameters():
        param.requires_grad = False
    for param in model.out.parameters():
        param.requires_grad = True
    # for param in model.out_activation.parameters():
        # param.requires_grad = True
    
    optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
    # This loss function applies sigmoid to output first to constrain between 0 and 1
    loss = torch.nn.BCEWithLogitsLoss()
    # loss = torch.nn.CrossEntropyLoss()
    # loss = torch.nn.MSELoss()
    
    # t = tqdm.trange(NUM_EPOCHS, leave=True)
    t = range(NUM_EPOCHS)
    # rp = mlvt.Reprint()
    # loss_plot = mlvt.Line(NUM_EPOCHS, 20, accumulate=NUM_EPOCHS, color="bright_green")
    model.train()
    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    # loader = DataLoader(train_dataset, batch_size=batch, sampler=sampler, num_workers=16, pin_memory=True)
    for epoch in t:
        correct = 0.0
        total = 0.0
        for i, batch in enumerate(train_loader):
            ids = torch.as_tensor(batch['input_ids'], dtype=torch.long).clone().detach()
            masks = torch.as_tensor(batch['attention_mask'], dtype=torch.long).clone().detach()
            token_type_ids = torch.as_tensor(batch['token_type_ids'], dtype=torch.long).clone().detach()
            targets = torch.as_tensor(batch['targets']).clone().detach()
            targets = targets.unsqueeze(1)

            # print(torch.unique(targets))
            ids, masks, token_type_ids, targets = ids.to(device), masks.to(device), token_type_ids.to(device), targets.to(device)
            
            optimizer.zero_grad()
            Y_hat = model(ids, masks, token_type_ids)

            with torch.no_grad():
                pred = torch.sigmoid(Y_hat).squeeze().cpu().detach().numpy()
                pred[pred>=config["prob_threshold"]] = 1
                pred[pred<config["prob_threshold"]] = 0
                correct += np.sum(pred == targets.cpu().numpy().squeeze())                
                total += len(targets)
                train_accuracy = correct / total

            l = loss(Y_hat, targets)
            l.backward()
            optimizer.step()
            # t.set_description(f"loss: {l.item():.3e} epoch: {epoch}")
            # if i % 100 == 0:
            #     loss_plot.update(l.item())
            #     rp.print(loss_plot)
            #     rp.flush()
        if epoch % 10 == 0:
            print(f'{epoch} Epochs Trained')
        print("Train accuracy: ", train_accuracy)
        my_lr_scheduler.step()
                
    # torch.save(model.state_dict(), "chkpoint.pt")
    model.eval()
    diff = 0
    tot = 0
    for batch in test_loader:
        ids = torch.as_tensor(batch['input_ids'], dtype=torch.long).clone().detach()
        masks = torch.as_tensor(batch['attention_mask'], dtype=torch.long).clone().detach()
        token_type_ids = torch.as_tensor(batch['token_type_ids'], dtype=torch.long).clone().detach()
        targets = torch.as_tensor(batch['targets']).clone().detach()
        # targets = targets.unsqueeze(1)
        # print(torch.unique(targets))
        ids, masks, token_type_ids, targets = ids.to(device), masks.to(device), token_type_ids.to(device), targets.to(device)
        Y_hat = model(ids, masks, token_type_ids)
        num_examples = len(targets)
        # pred = torch.round(torch.sigmoid(Y_hat)).squeeze().cpu().detach().numpy()
        pred = torch.sigmoid(Y_hat).squeeze().cpu().detach().numpy()
        pred[pred>=config["prob_threshold"]] = 1
        pred[pred<config["prob_threshold"]] = 0
        targ = targets.cpu().detach().numpy().astype(int)
        num_right = np.sum(pred.astype(int) == targets)
        
        metrics_dict = calculate_metrics(targ, pred, torch.sigmoid(Y_hat).squeeze().cpu().detach().numpy())

        # tune.report(score=metrics_dict['f1'])
        
        d = num_examples - num_right
        diff += d
        tot += num_examples
        # print(f"{(tot - diff) / tot * 100}% accuracy on test ({diff} wrong of {tot})")

    return metrics_dict, train_accuracy

def calculate_metrics(targets, pred, pred_not_int):
    metrics = {}
    metrics['conf_matrix'] = confusion_matrix(y_true=targets, y_pred=pred)
    metrics['precision']   = precision_score(targets, pred)
    metrics['recall']      = recall_score(targets, pred)
    metrics['f1']          = f1_score(targets, pred)
    metrics['accuracy']    = accuracy_score(targets, pred)
    metrics['roc_curve']   = roc_curve(targets, pred_not_int)
    metrics['auroc']       = roc_auc_score(targets, pred_not_int)

    print(repr(metrics))

    return metrics

if __name__ == "__main__":

    one_config = {
        "lr": 1e-3,
        "batch_size": 256,
        "prob_threshold": 0.5,
        "optimizer": torch.optim.Adam,
        "max_len": 256,
    }

    # ray.init(num_gpus=1)

    metrics = {}
    train_acc = {}

    fig, ax = plt.subplots()

    names = ['BERT_Supervised', 'BERT_Weak', 'ClinicalBERT_Supervised', 'ClinincalBERT_Weak']

    for i, model in enumerate(["bert-base-uncased", "emilyalsentzer/Bio_ClinicalBERT"]):
        for j, labeling in enumerate(["supervised", "weak"]):
            print(f'Running {names[2*i+j]}')
            if labeling == "weak":
                config = {"lr": 1e-2, 
                "batch_size": 128,
                "prob_threshold": 0.6,
                "optimizer": torch.optim.Adam,
                "max_len": 256}
            else:
                config = one_config

            metrics[names[2*i+j]], train_acc[names[2*i+j]] = train_model(config, model, labeling)

            ax.plot(metrics[names[2*i+j]]['roc_curve'][0], metrics[names[2*i+j]]['roc_curve'][1], label=names[2*i+j])
            ax.set_xlabel('FP Rate')
            ax.set_ylabel('TP Rate')
            ax.legend()

            fig1, ax1 = plt.subplots(figsize=(5, 5))
            ax1.matshow(metrics[names[2*i+j]]['conf_matrix'], cmap=plt.cm.Oranges, alpha=0.3)
            for ii in range(metrics[names[2*i+j]]['conf_matrix'].shape[0]):
                for jj in range(metrics[names[2*i+j]]['conf_matrix'].shape[1]):
                    ax1.text(x=jj, y=ii,s=metrics[names[2*i+j]]['conf_matrix'][ii, jj], va='center', ha='center', size='xx-large')
            
            ax1.set_xlabel('Predictions', fontsize=18)
            ax1.set_ylabel('Actual', fontsize=18)
            ax1.set_title('Confusion Matrix', fontsize=18)
            fig1.savefig('conf_mat'+'_'+names[2*i+j]+'.png', dpi=300)

    fig.savefig('roc_curve.png')

    for key in names:
        print('============================================\n\n\n\n')
        print(key+ '\n\n\n')
        print(metrics[key])
                    
    # analysis = tune.run(train_model, config=config, resources_per_trial={"cpu": 8, "gpu": 1})
    # print("Best config: ", analysis.get_best_config(metric="score", mode="max"))
    # print("Best metric: ", analysis.get_best_trial(metric="score", mode="max"))

    # df = analysis.dataframe()
    # print(df)
    
    
