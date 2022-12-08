from transformers import AutoTokenizer, AutoModel
import transformers
from util.data_cleaning import load_data, weighted_sampler
from util.dataset import BertDataset
import torch
from model.model import BertPretrained
import mlvt
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

NUM_EPOCHS = 100

def train_model():
    batch_size = 128
    max_len = 128
    pretrained_model = 'bert-base-uncased'
    # pretrained_model = "emilyalsentzer/Bio_ClinicalBERT" 

    # X_train, X_test, y_train, y_test = load_data("data/food_insecurity_labels_v2.csv")
    # X_train, X_test, y_train, y_test = load_data("data/food_insecurity_labels_v2.csv",
    #                                              unstructured_data="data/unstructured_data.csv") # weak labeling
    X_train, X_test, y_train, y_test = load_data("data/food_insecurity_labels_v2.csv",
                                                 unstructured_data="data/raw_weak_labels.csv") # weak labeling                    
    print(f'number training examples: {len(X_train)}')
    print(f'number of labels: {len(y_train)}')
    print(f'number positive training labels: {np.sum(y_train)}')
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
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)
    # This loss function applies sigmoid to output first to constrain between 0 and 1
    loss = torch.nn.BCEWithLogitsLoss()
    # loss = torch.nn.CrossEntropyLoss()
    # loss = torch.nn.MSELoss()
    
    t = tqdm.trange(NUM_EPOCHS, leave=True)
    rp = mlvt.Reprint()
    loss_plot = mlvt.Line(NUM_EPOCHS, 20, accumulate=NUM_EPOCHS, color="bright_green")
    model.train()
    # loader = DataLoader(train_dataset, batch_size=batch, sampler=sampler, num_workers=16, pin_memory=True)
    for epoch in t:
        correct = 0
        total = 0
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
            # print(Y_hat)
            # print(targets)
            l = loss(Y_hat, targets)
            l.backward()
            optimizer.step()
            t.set_description(f"loss: {l.item():.3e} epoch: {epoch}")
            if i % 100 == 0:
                loss_plot.update(l.item())
                rp.print(loss_plot)
                rp.flush()
                
    torch.save(model.state_dict(), "chkpoint.pt")
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
        pred = torch.round(torch.sigmoid(Y_hat)).squeeze()
        num_right = sum(pred == targets)
        
        metrics_dict = calculate_metrics(targets.cpu().detach().numpy(), pred.cpu().detach().numpy())
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(metrics_dict['conf_matrix'], cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(metrics_dict['conf_matrix'].shape[0]):
            for j in range(metrics_dict['conf_matrix'].shape[1]):
                ax.text(x=j, y=i,s=metrics_dict['conf_matrix'][i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.savefig('conf_mat.png', dpi=300)
        d = num_examples - num_right
        diff += d
        tot += num_examples
        print(f"{(tot - diff) / tot * 100}% accuracy on test ({diff} wrong of {tot})")

def calculate_metrics(targets, pred):
    metrics = {}
    print(targets)
    print(pred)
    print(targets - pred)
    metrics['conf_matrix'] = confusion_matrix(y_true=targets, y_pred=pred)
    metrics['precision']   = precision_score(targets, pred)
    metrics['recall']      = recall_score(targets, pred)
    metrics['f1']          = f1_score(targets, pred)
    metrics['accuracy']    = accuracy_score(targets, pred)

    print(repr(metrics))

    return metrics
        
                
train_model()
    
    
