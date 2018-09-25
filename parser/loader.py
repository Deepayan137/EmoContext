from config import *
import pdb
from tqdm import *
from collections import defaultdict

def parse(**kwargs):
	path = kwargs['path']
	with open(path, 'r') as f:
		lines = f.readlines()
	
	data = defaultdict(lambda: defaultdict(int))
	for i, line in enumerate(tqdm(lines[1:])):
		units = line.split('\t')
		index = int(units[0])
		text = ' '.join(units[1:4])
		data[index]['text'] = text
		data[index]['label'] = units[4].strip()
	pdb.set_trace()
parse(path='data/train.txt')