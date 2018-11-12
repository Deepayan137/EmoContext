import os
import torch
import logging
import pdb
import collections

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config.opts import Config
from model.model import *
from model.model_attn import *
from utils.loader import *
from utils.utils import *
  
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_sep(**kwargs):
    input_ = kwargs['input']
    checkpoint = kwargs['checkpoint']
    savepath = kwargs['savepath']
    start_epoch = checkpoint['epoch']
    model = kwargs['model'].to(device)
    lmap = kwargs['lamp']
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    # pdb.set_trace()
    eval_object = Eval(lmap)

    train_subset, val_subset = split(input_, split=0.8, random=True)
    
    train_data_loader = DataLoader(
                                dataset=train_subset,
                                batch_size=32,
                                shuffle=True)
    val_data_loader = DataLoader(
                            dataset=val_subset,
                            batch_size=32,
                            shuffle=False)

    loader = {'train':train_data_loader, 'val':val_data_loader}
    keys = ['train', 'val']
    for epoch in range(start_epoch, kwargs['epochs']):
        print('Epochs:[%d]/[%d]'%(epoch, kwargs['epochs']))
        for key in keys:
            avgLoss = AverageMeter("{} loss".format(key))
            avgF1 = AverageMeter("{} F1".format(key))
            for iteration, batch in enumerate(tqdm(loader[key])):
                
                input_feature = [batch['input'][i].to(device) for i in range(len(batch['input']))]
                target = batch['target'].to(device)
                # pdb.set_trace()
                output = model(input_feature)
                # pdb.set_trace()
                prediction = output.contiguous()
                loss = criterion(prediction, target)
                f1 = eval_object.f1(prediction, target)
                avgLoss.add(loss.item())
                avgF1.add(f1)
                if key != 'val':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print('\n {} set: loss: {:.2f} F1: {:.2f} \n'.format(key, avgLoss.compute(), 
                                                                avgF1.compute()))
        info = '%d %.2f %.2f \n'%(epoch, avgLoss.compute(), avgF1.compute())
        logging.info(info)
        state = avgLoss.compute()
        print(checkpoint['best'])
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best': state
                        }, savepath,
                        is_best)

def main(**kwargs):
    opt = Config()
    opt._parse(kwargs)
    
    path = opt.path
    lmap = opt.lmap
    vector_size = '%dd'%opt.inp
    print(vector_size)
    splits = ['train', 'test']
    datasets = {} 
    # for split in splits:
    #   datasets[split] = TweetData(path,split,lmap, vector_size = vector_size)
    # datasets['train'] = pad_seq(datasets['train'])
    
    # pdb.set_trace()
    epochs = opt.epochs
    nIn = opt.inp
    nHidden = opt.hidden    
    nClasses = opt.out
    depth = opt.depth
    filters = opt.filters
    seqlen = 156
    datasets = TweetData_V02('data/train.txt', lmap, nIn)
    model = RNN_attn(nIn, nHidden, nClasses, depth).to(device)
    # model = RCNN(nIn, nHidden, nClasses, seqlen, filters)
    # model = Turnip(nIn, nHidden, nClasses, depth)
    save_dir = opt.save_dir
    # gmkdir(save_dir)
    save_file = opt.save_file
    savepath = save_dir + '/' + save_file 
    
    if os.path.isfile(savepath):
        checkpoint = torch.load(savepath)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(savepath))
        checkpoint = {
                "epoch":0,
                "best":float("inf")
                }
    
    lr = opt.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gmkdir('logs')
    logging.basicConfig(filename="logs/%s.log"%save_file, level=logging.INFO)

    train_sep(input=datasets,
    model=model,
    lamp=lmap,
    epochs=epochs,
    checkpoint=checkpoint,
    savepath=savepath,
    optimizer=optimizer,
    criterion=criterion)


    


if __name__ == '__main__':
    import fire
    fire.Fire(main)