import torch
import numpy as np
import torch.optim as optim
import warnings
from args import get_parse
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import random
import os
import torch.nn.functional as F
from utils import accuracy
from dataset.dataset import AllData_DataFrame
from torch.utils.data import DataLoader
from models.MGN import MGN

warnings.filterwarnings("ignore")
seed = int(1111)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

args = get_parse()

epochs = 400
stop_epochs = 200
batch_size = 8

env_name = f"test"
use_model = MGN

accs = []
aucs = []
sens = []
spes = []

early_stop = 0

corrs = []
try:
    os.makedirs(f"checkpoints/{env_name}")
except:
    pass

save_pred = []
save_label = []

for fold in range(10):
    print("Conducting... ",fold)
    args.fold = fold

    train_data = AllData_DataFrame(args, is_train = True)
    test_data = AllData_DataFrame(args, is_test = True)
    val_data = AllData_DataFrame(args, is_val = True)

    train_loader = DataLoader(train_data,8,shuffle=True,num_workers=0)
    test_loader = DataLoader(test_data,8,shuffle=True,num_workers=0)
    val_loader = DataLoader(val_data,8,shuffle=False,num_workers=0)
    
    in_channel = 2
    kernel = 100

    smodel = use_model(in_channel=in_channel,kernel_size=kernel,num_classes = 2, args = args)

    optimizer = optim.Adam(smodel.parameters(),lr=3e-4, weight_decay=0.00005)

    smodel.cuda()

    res = {
    "best_acc": 0.,
    "best_auc": 0.,
    "best_sen": 0.,
    "best_spe": 0.,
    }

    pred_all = []
    label_all = []

    train_best_acc = -1.
    for epoch in range(epochs):
        smodel.train()
        for idx, (img1, img2, label) in enumerate(train_loader):
            img1 = img1.float().cuda()
            img2 = img2.float().cuda()
            label = label.long().cuda()
            optimizer.zero_grad()
            alpha_t = args.alpha * ((epoch + 1) / 50)

            output1,log1,log2 = smodel(img1, img2)
            loss_train = F.cross_entropy(output1, label) + alpha_t * F.cross_entropy(log1, label) + alpha_t * F.cross_entropy(log2, label)

            loss_train.backward()
            optimizer.step()

            # TODO
            # save best model By val data
    # after training

    smodel.eval()
    count = 0.
    acc_score = 0.
    loss_score = 0.
    test_pred = []
    test_label = []
    for _, (img1, img2, label) in enumerate(val_loader):
        img1 = img1.float().cuda()
        img2 = img2.float().cuda()
        label = label.long().cuda()
        with torch.no_grad():
            pred,_,_ = smodel(img1, img2)
            loss_val = F.cross_entropy(pred, label).cpu().detach().numpy()
            acc_val = accuracy(pred,label).cpu().detach().numpy()
            count += len(img1)
            acc_score += acc_val * len(img1)
            loss_score += loss_val * len(img1)

            test_pred.extend(pred.max(1)[1].cpu().detach().numpy())
            test_label.extend(label.cpu().detach().numpy())
    acc_score /= count
    loss_score /= count

    print(f"Training {epoch}: Acc-{acc_score}")
    train_best_acc = acc_score
    pred_label_cpu = test_pred
    test_label_cpu = test_label

    pred_all = np.array(pred_label_cpu)
    label_all = np.array(test_label_cpu)
    cm = confusion_matrix(pred_all.ravel(),label_all.ravel())
    acc = accuracy_score(pred_all,label_all)
    auc_val = f1_score(pred_all,label_all,average='micro')
    eval_sen = round(cm[1, 1] / float(cm[1, 1]+cm[1, 0]),4)
    eval_spe = round(cm[0, 0] / float(cm[0, 0]+cm[0, 1]),4)
    
    res['best_acc']  = acc
    res['best_auc']  = auc_val
    res['best_sen']  = eval_sen
    res['best_spe']  = eval_spe

    print('Epoch: {:04d}'.format(epoch+1),
        'acc_val: {:.4f}'.format(acc),
        'eval_sen: {:.4f}'.format(eval_sen),
        'eval_spe: {:.4f}'.format(eval_spe),
        'auc_val: {:.4f}'.format(auc_val))

    accs.append(res['best_acc'])
    aucs.append(res['best_auc'])

    sens.append(res['best_sen'])
    spes.append(res['best_spe'])

    print(pred_all, label_all)

    save_label.append(label_all)
    save_pred.append(pred_all)
sens = [f for f in sens if f == f]
spes = [f for f in spes if f == f]

print(accs)
print("ACC-%.4f %.4f" % (np.mean(accs) * 100,np.std(accs) * 100) )
print("Sen-%.4f %.4f" % (np.mean(sens) * 100,np.std(sens) * 100) )
print("F1-%.4f %.4f" % (np.mean(aucs) * 100,np.std(aucs) * 100) )