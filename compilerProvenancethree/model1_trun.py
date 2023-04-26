import time

import torch
import torch as th
from torch import optim, nn

from train_evalcnn import model_train1, model_evaluate1, metric_predictions1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model1_trun(fig_prefix,model,trainset,testset):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)
    start = time.time()
    model_stats, history = model_train1(fig_prefix, trainset, model, loss_func, optimizer, scheduler, testset)
    end = time.time()
    # step 5: evaluate the trained model on the test data
    print('evaluating on the test-set the trained -textcnn model')
    # model = init_model.load_state_dict(th.load(model_stats)).to(device)
    model.load_state_dict(th.load(model_stats))
    model.eval()

    preds, ground_truth = model_evaluate1(model, testset, loss_func, fig_prefix)
    return preds, ground_truth
    #metric_predictions1(preds, ground_truth, fig_prefix)
    #th.cuda.empty_cache()