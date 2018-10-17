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
def write_(labels):
	with open('data/test.txt', 'r') as f:
		lines = f.readlines()
	
	with open('data/test_.txt', 'a') as f:
		for i, line in enumerate(lines[1:]):
			line_new = line.strip() + '\t' + labels[i] + '\n'
			f.write(line_new)
	


def test(**kwargs):
	input_ = kwargs['input']
	model = kwargs['model']
	lmap = kwargs['lmap']
	eval_object = Eval(lmap)
	results = []
	ilmap = {v:k for k,v in lmap.items()}
	for idx in trange(len(input_)):

		sequence = torch.from_numpy(input_[idx]['feature']).to(device)
		text = input_[idx]['input']
		# sequence = sequence.permute(1, 0)
		sequence = torch.unsqueeze(sequence, 0)
		output = model(sequence)
		prediction = output.contiguous()
		prediction = eval_object.decode(prediction)
		results.append(ilmap[prediction.cpu().numpy()[0]])
	write_(results)

def test_batch(**kwargs):
	input_ = kwargs['input']
	model = kwargs['model'].to(device)
	lmap = kwargs['lamp']
	ilmap = {v:k for k,v in lmap.items()}
	# pdb.set_trace()
	eval_object = Eval(lmap)
	test_data_loader = DataLoader(dataset=input_,
								batch_size=32)
	results = []
	for iteration, batch in enumerate(tqdm(test_data_loader)):
		input_feature = batch['feature'].to(device)
		pdb.set_trace()
		output = model(input_feature)
		prediction = output.contiguous()
		prediction = eval_object.decode(prediction)
		results.extend(ilmap[prediction.cpu().numpy()[0]])
	write_(results)

def main(**kwargs):
	opt = Config()
	opt._parse(kwargs)
	path = opt.path
	lmap = opt.lmap
	vector_size = '%dd'%opt.inp
	datasets = {} 
	datasets['test'] = TweetData(path,'test',lmap, vector_size=vector_size)
	nIn = opt.inp
	nHidden = opt.hidden	
	nClasses = opt.out
	depth = opt.depth
	filters = opt.filters
	seqlen = 156
	model = RCNN_Text(nIn, nHidden).to(device)
	# model = RCNN(nIn, nHidden, nClasses, seqlen, filters).cuda()
	# model = EmoNet(nIn, nHidden, nClasses, depth).to(device)
	save_dir = opt.save_dir
	# gmkdir(save_dir)
	save_file = opt.save_file
	savepath = save_dir + '/' + save_file
	checkpoint = torch.load(savepath)
	model.load_state_dict(checkpoint['state_dict'])
	test(input=datasets['test'].data,
		model=model,
		lmap=lmap)

if __name__ == '__main__':
	import fire
	fire.Fire(main)
