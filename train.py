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


def train_subroutine(pair, model, optimizer, criterion, eval_object, validation=True):
	sequence, label = pair
	sequence = sequence.to(device)
	label = label.to(device)
	model = model.cuda()
	# sequence = sequence.view(1, *sequence.size())
	sequence = torch.unsqueeze(sequence, 1)
	label = torch.unsqueeze(label,0)
	output = model(sequence)
	prediction = output.contiguous()
	loss = criterion(prediction, label)
	

	if not validation:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	return loss.item(), eval_object.decode(prediction)

# data_loader  = DataLoader(dataset=input_,
# 									batch_size=64,
# 									shuffle=True)
def train(**kwargs):
	input_ = kwargs['input']
	checkpoint = kwargs['checkpoint']
	savepath = kwargs['savepath']
	start_epoch = checkpoint['epoch']
	model = kwargs['model']
	lmap = kwargs['lamp']

	eval_object = Eval(lmap)
	train_subset, val_subset = split(input_, split=0.8, random=True)

	for epoch in range(start_epoch, kwargs['epochs']):
		avgTrain = AverageMeter("train loss")
		epoch = epoch + 1 
		print('Epochs:[%d]/[%d]'%(epoch, kwargs['epochs']))
		train_pred =[]
		# pdb.set_trace()
		for idx in trange(len(train_subset)):
			input_feature = train_subset[idx]['feature']
			target = train_subset[idx]['target']
			pair = (torch.from_numpy(input_feature), torch.tensor(target)) 
			loss, prediction = train_subroutine(pair, model, kwargs['optimizer'], kwargs['criterion'], eval_object, validation=False)
			avgTrain.add(loss)
			train_pred.append(prediction)
		train_loss = avgTrain.compute()
		train_f1 = eval_object.f1(input_, train_pred, train_subset)
		print('\nTrain set: Average loss: {}, F1: ({})\n'.format(train_loss, train_f1))

		avgValidation = AverageMeter("validation loss")
		val_pred = []
		for idx in trange(len(val_subset)):
			input_feature = val_subset[idx]['feature']
			target = val_subset[idx]['target']
			pair = (torch.from_numpy(input_feature), torch.tensor(target)) 
			loss, prediction = train_subroutine(pair, model, kwargs['optimizer'], kwargs['criterion'], eval_object)
			val_pred.append(prediction)
			avgValidation.add(loss)
		validation_loss = avgValidation.compute()
		val_f1 = eval_object.f1(input_, val_pred, val_subset)
		# validation_accuracy = avgAcc.compute()
		print('\nValidation set: Average loss: {}, F1: ({})\n'.format(validation_loss, val_f1))

		info = '%d %.2f %.2f %.2f %.2f\n'%(epoch, train_loss, validation_loss, train_f1, val_f1)
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

def train_batch(**kwargs):
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
				input_feature = batch['feature'].to(device)
				target = batch['target'].to(device)
				output = model(input_feature)
				pdb.set_trace()
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


def train_sepTurn(**kwargs):
	input_ = kwargs['input']
	checkpoint = kwargs['checkpoint']
	savepath = kwargs['savepath']
	start_epoch = checkpoint['epoch']
	model = kwargs['model'].to(device)
	lmap = kwargs['lamp']
	criterion = kwargs['criterion']
	optimizer = kwargs['optimizer']
	nIn = kwargs['nIn']

	eval_object = Eval(lmap)
	train_subset, val_subset = split(input_, split=0.8, random=True)

	# pdb.set_trace()

	for epoch in range(start_epoch, kwargs['epochs']):

		print('Epochs:[%d]/[%d]'%(epoch, kwargs['epochs']))
		train_pred =[]
		avgTrain = AverageMeter("train loss")
		#actual_samples = 0
		for idx in trange(len(train_subset)):

			x1 = torch.from_numpy(train_subset[idx]['feature_turn1']).view(-1, 1, nIn).to(device)
			x2 = torch.from_numpy(train_subset[idx]['feature_turn2']).view(-1, 1, nIn).to(device)
			x3 = torch.from_numpy(train_subset[idx]['feature_turn3']).view(-1, 1, nIn).to(device)

			target = torch.tensor([train_subset[idx]['target']]).to(device)

			try:
				out = model(x1, x2, x3)
				loss = criterion(out, target)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				avgTrain.add(loss)
				train_pred.append(out)
				#actual_samples += 1
			except:
				pass

			

		train_loss = avgTrain.compute()
		# train_f1 = eval_object.f1(input_, train_pred, train_subset)
		# print('\nTrain set: Average loss: {}, F1: ({})\n'.format(train_loss, train_f1))
		print('\nTrain set: Average loss: {}\n'.format(train_loss))

		avgValidation = AverageMeter("validation loss")
		val_pred = []
		for idx in trange(len(val_subset)):

			x1 = torch.from_numpy(val_subset[idx]['feature_turn1']).view(-1, 1, nIn).to(device)
			x2 = torch.from_numpy(val_subset[idx]['feature_turn2']).view(-1, 1, nIn).to(device)
			x3 = torch.from_numpy(val_subset[idx]['feature_turn3']).view(-1, 1, nIn).to(device)

			target = torch.tensor([val_subset[idx]['target']]).to(device)

			try:
				out = model(x1, x2, x3)
				loss = criterion(out, target)

				val_pred.append(out)
				avgValidation.add(loss)
			except:
				pass
			
		validation_loss = avgValidation.compute()
		# val_f1 = eval_object.f1(input_, val_pred, val_subset)
		# validation_accuracy = avgAcc.compute()
		# print('\nValidation set: Average loss: {}, F1: ({})\n'.format(validation_loss, val_f1))
		print('\nValidation set: Average loss: {}\n'.format(validation_loss))

		## TO DO: F1 Score, log and Saving Checkpoint ##



def main(**kwargs):
	opt = Config()
	opt._parse(kwargs)
	
	path = opt.path
	lmap = opt.lmap
	vector_size = '%dd'%opt.inp
	print(vector_size)
	splits = ['train', 'test']
	datasets = {} 
	for split in splits:
		datasets[split] = TweetData(path,split,lmap, vector_size = vector_size)
	# datasets['train'] = pad_seq(datasets['train'])

	
	epochs = opt.epochs
	nIn = opt.inp
	nHidden = opt.hidden	
	nClasses = opt.out
	depth = opt.depth
	filters = opt.filters
	seqlen = 156
	# model = WordEncoder(nIn, nHidden, nClasses, depth).to(device)
	model = RCNN(nIn, nHidden, nClasses, seqlen, filters)
	# model = RCNN_Text(nIn, nHidden)

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

	# train_batch(input=datasets['train'],
	# model=model,
	# lamp=lmap,
	# epochs=epochs,
	# checkpoint=checkpoint,
	# savepath=savepath,
	# optimizer=optimizer,
	# criterion=criterion)

	train_sepTurn(input=datasets['train'],
	model=model,
	lamp=lmap,
	epochs=epochs,
	checkpoint=checkpoint,
	savepath=savepath,
	optimizer=optimizer,
	criterion=criterion,
	nIn=nIn)

	


    


if __name__ == '__main__':
	import fire
	fire.Fire(main)

	