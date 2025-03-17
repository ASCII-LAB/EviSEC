import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import utils as u
import torch
from torch_geometric.utils import stochastic_blockmodel_graph
from tqdm import tqdm
from .sbm_seed import stochastic_blockmodel_graph_with_seed

class Brain_Dataset():
	def __init__(self,args, ood_mode=None):  
		if not isinstance(args.brain_data_args, u.Namespace):
			args.brain_data_args = u.Namespace(args.brain_data_args)
		self.num_classes = 10	
		data = np.load(os.path.join(args.brain_data_args.folder,args.brain_data_args.file))
		self.nodes_labels_times = self.load_node_labels(data) 
		self.edges = self.load_transactions(data)
		self.num_nodes, self.nodes_feats = self.load_node_feats(data)
		self.num_non_existing = 12* self.num_nodes ** 2 - len(self.edges)
		
		if ood_mode == "SM":  # SM FI
			n = self.num_nodes
			d = self.edges['idx'].shape[0]/self.num_nodes/(self.num_nodes-1)/(self.max_time-self.min_time+1)
			num_blocks = self.num_classes
			p_ii, p_ij = 0.5 * d, 1.5 * d  
			block_size = n // num_blocks
			block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size] 
			edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
			edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii 
			
			SM_edges = []
			save_path = f"./DL/data_SM/{args.data}_edges.pt"
			if os.path.exists(save_path):
				self.edges = torch.load(save_path)
				print(f"Loaded existing edges from {save_path}")
			else:	
				for timestep in tqdm(range(self.min_time, int(self.max_time) + 1), desc=f"Generating SM Graphs for {args.data}"):

					edge_index = stochastic_blockmodel_graph_with_seed(block_sizes, edge_probs,time_seed=timestep)  
					cols_time = torch.tensor([timestep] * edge_index.size(1))
					edge_index_time = torch.cat([edge_index.T, cols_time.unsqueeze(1)], dim=1)
					SM_edges.append(edge_index_time)
					
					
				self.edges["idx"] = torch.cat(SM_edges, dim=0)
				self.edges["vals"] = torch.ones(self.edges["idx"].size(0))
				torch.save(self.edges, save_path)
				print(f"Edges saved to {save_path}")

		elif ood_mode == "FI":
			FI_nodes_feats = []
			for timestep in range(self.min_time,int(self.max_time)+1):
				x = self.nodes_feats[timestep] # [5000, 20]
				n = self.num_nodes
				idx = torch.randint(0, n, (n, 2)) 
				weight = torch.rand(n).unsqueeze(1) 
				x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)
				FI_nodes_feats.append(x_new)
			print(f"FI_nodes_feats{len(FI_nodes_feats)}*{FI_nodes_feats[0].shape}:::{FI_nodes_feats[0][:3]}")
			self.nodes_feats = FI_nodes_feats


	def load_node_feats(self, data):
		features = data['attmats']
		nodes_feats = []
		for i in range(12):
			nodes_feats.append(torch.FloatTensor(features)[:, i]) 

		self.num_nodes = 5000
		print(self.num_nodes)
		self.feats_per_node = len(nodes_feats[0])

		return self.num_nodes, nodes_feats  


	def load_node_labels(self, data):
		lcols = u.Namespace({'nid': 0,
							 'label': 1}) 


		labels = data['labels']

		nodes_labels_times =[]
		for i in range(len(labels)):
			label = labels[i].tolist().index(1)
			nodes_labels_times.append([i, label])

		nodes_labels_times = torch.LongTensor(nodes_labels_times)

		return nodes_labels_times

	def load_transactions(self, data):
		adj = data['adjs']

		tcols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})
		
		data = []

		t = 0
		for graph in adj:
			for i in range(len(graph)):
				temp = np.concatenate((np.ones(len(np.where(graph[i] == 1)[0])).reshape(-1,1)*i,np.where(graph[i] == 1)[0].reshape(-1,1), np.ones(len(np.where(graph[i] == 1)[0])).reshape(-1,1) * t) , 1).astype(int).tolist()
				data.extend(temp)
			t += 1

		data= torch.LongTensor(data)

		self.max_time = torch.FloatTensor([11])
		self.min_time = 0

		print(data.size(0))

		return {'idx': data, 'vals': torch.ones(data.size(0))}
