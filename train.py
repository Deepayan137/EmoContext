import os
import torch
import logging
import pdb

from torch import nn, optim
from torch.autograd import Variable

from config.opts import Config
from model.model import *
from utils.loader import *
from utils.utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_subroutine(pair, model, optimizer, criterion, lmap, validation=True):
	sequence, label = pair
	sequence = sequence.to(device)
	label = label.to(device)
	# sequence = sequence.view(1, *sequence.size())
	sequence = torch.unsqueeze(sequence, 1)
	label = torch.unsqueeze(label,0)
	output = model(sequence)
	prediction = output.contiguous()
	loss = criterion(prediction, label)
	eval_object = Eval(lmap)

	if not validation:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	return loss.item(), eval_object.predict(prediction, label)


def train(**kwargs):
	input_ = kwargs['input']
	checkpoint = kwargs['checkpoint']
	savepath = kwargs['savepath']
	start_epoch = checkpoint['epoch']
	model = kwargs['model']
	lmap = kwargs['lamp']
	# pdb.set_trace()
	train_subset, val_subset = split(input_, split=0.8, random=False)
	for epoch in range(start_epoch, kwargs['epochs']):
		avgTrain = AverageMeter("train loss")
		avgAcc = AverageMeter("accuracy")
		epoch = epoch + 1 
		print('Epochs:[%d]/[%d]'%(epoch, kwargs['epochs']))
		for idx in trange(len(train_subset)):
			input_feature = input_[idx]['feature']
			target = input_[idx]['target']
			pair = (torch.from_numpy(input_feature), torch.tensor(target)) 
			loss, prediction = train_subroutine(pair, model, kwargs['optimizer'], kwargs['criterion'], lmap, validation=False)
			avgTrain.add(loss)
			avgAcc.add(prediction)
		train_loss = avgTrain.compute()
		train_accuracy = avgAcc.compute()
		print('\nTrain set: Average loss: {}, Accuracy: ({})\n'.format(train_loss, train_accuracy))

		avgValidation = AverageMeter("validation loss")
		avgAcc = AverageMeter("accuracy")
		for idx in trange(len(val_subset)):
			input_feature = input_[idx]['feature']
			target = input_[idx]['target']
			pair = (torch.from_numpy(input_feature), torch.tensor(target)) 
			loss, prediction = train_subroutine(pair, model, kwargs['optimizer'], kwargs['criterion'], lmap)
			avgValidation.add(loss)
			avgAcc.add(prediction)
		validation_loss = avgValidation.compute()
		validation_accuracy = avgAcc.compute()
		print('\nValidation set: Average loss: {}, Accuracy ({})\n'.format(validation_loss, validation_accuracy))

		info = '%d %.2f %.2f %.2f %.2f\n'%(epoch, train_loss, validation_loss, train_accuracy, validation_accuracy)
		logging.info(info)
		state = validation_loss
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
		
	print("finished training ...")


def main(**kwargs):
	opt = Config()
	opt._parse(kwargs)
	
	path = opt.path
	lmap = opt.lmap
	splits = ['train', 'test']
	datasets = {} 
	for split in splits:
		datasets[split] = TweetData(path,split,lmap)
	
	epochs = opt.epochs
	nIn = opt._in
	nHidden = opt.hidden	
	nClasses = opt.out
	depth = opt.depth
	
	model = EmoNet(nIn, nHidden, nClasses, depth).to(device)
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

	train(input=datasets['train'].data,
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

	