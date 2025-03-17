import os
import sys
sys.path.append(os.path.abspath('..'))
import utils as u
import torch
#erase
import time
import tarfile
import itertools
import numpy as np
import traceback
from torch_geometric.utils import stochastic_blockmodel_graph
from .sbm_seed import stochastic_blockmodel_graph_with_seed
from tqdm import tqdm

class Elliptic_Temporal_Dataset():
	def __init__(self,args, ood_mode=None):
		if not isinstance(args.elliptic_args, u.Namespace):
			args.elliptic_args = u.Namespace(args.elliptic_args)

		tar_file = os.path.join(args.elliptic_args.folder, args.elliptic_args.tar_file)
		tar_archive = tarfile.open(tar_file, 'r:gz')
		self.nodes_labels_times = self.load_node_labels(args.elliptic_args, tar_archive)
		self.edges = self.load_transactions(args.elliptic_args, tar_archive) 
		self.nodes, self.nodes_feats = self.load_node_feats(args.elliptic_args, tar_archive)


		if ood_mode == "SM":  # SM FI LO
			n = self.num_nodes
			d = self.edges['idx'].shape[0]/self.num_nodes/(self.num_nodes-1)/(self.max_time-self.min_time+1)
			num_blocks = 2
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
				for timestep in tqdm(range(int(self.min_time), int(self.max_time) + 1), desc=f"Generating SM Graphs for {args.data}"):
					edge_file_path = f"./DL/data_SM/{args.data}_edges_SM_{timestep}.pt"
					if os.path.exists(edge_file_path):
						print(f"File {edge_file_path} already exists. Skipping.")
						SM_edges.append(torch.load(edge_file_path))
						continue  
					edge_index = stochastic_blockmodel_graph_with_seed(block_sizes, edge_probs,time_seed=timestep)  
					cols_time = torch.tensor([timestep] * edge_index.size(1))
					edge_index_time = torch.cat([edge_index.T, cols_time.unsqueeze(1)], dim=1)
					SM_edges.append(edge_index_time)
					torch.save(edge_index_time, edge_file_path)
				self.edges["idx"] = torch.cat(SM_edges, dim=0)
				self.edges["vals"] = torch.ones(self.edges["idx"].size(0))
				torch.save(self.edges, save_path)
				print(f"Edges saved to {save_path}")

		elif ood_mode == "FI":
			print("----- ID edges:",len(self.edges['idx'])) 
			print(f"self.nodes_feats.shape:{self.nodes_feats.shape}")
			print(f"self.num_nodes:{self.num_nodes}")
			print("--------------:",len(self.edges['idx'])) 
			x = self.nodes_feats
			n = self.num_nodes
			idx = torch.randint(0, n, (n, 2)) 
			weight = torch.rand(n).unsqueeze(1) 
			x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)			
			print(f"FI_nodes_feats{len(x_new)}*{x_new[0].shape}:::{x_new[0][:3]}")
			self.nodes_feats = x_new

	def load_node_feats(self, elliptic_args, tar_archive):
		try:
			data = u.load_data_from_tar(elliptic_args.feats_file, tar_archive, starting_line=0)
		except Exception as e:
			print('== Exception: ',e)
			traceback.print_exc()
			filepath = os.path.join(elliptic_args.folder, elliptic_args.feats_file)
			data = u.load_data_from_file(filepath, starting_line=0)
		nodes = data

		nodes_feats = nodes[:,1:]


		self.num_nodes = len(nodes)
		self.feats_per_node = data.size(1) - 1

		return nodes, nodes_feats.float()


	def load_node_labels(self, elliptic_args, tar_archive):
		try:
			labels = u.load_data_from_tar(elliptic_args.classes_file, tar_archive, replace_unknow=True).long()
		except Exception as e:
			print('== Exception: ',e)
			traceback.print_exc()
			filepath = os.path.join(elliptic_args.folder, elliptic_args.classes_file)
			labels = u.load_data_from_file(filepath, replace_unknow=True)
		try:
			times = u.load_data_from_tar(elliptic_args.times_file, tar_archive, replace_unknow=True).long()
		except Exception as e:
			print('== Exception: ',e)
			traceback.print_exc()
			filepath = os.path.join(elliptic_args.folder, elliptic_args.times_file)
			times = u.load_data_from_file(filepath, replace_unknow=True)
		lcols = u.Namespace({'nid': 0,
							 'label': 1})
		tcols = u.Namespace({'nid':0, 'time':1})


		nodes_labels_times =[]
		for i in range(len(labels)):
			label = labels[i,[lcols.label]].long()
			if label>=0:
				nid=labels[i,[lcols.nid]].long()
				time=times[nid,[tcols.time]].long()
				nodes_labels_times.append([nid , label, time])
		nodes_labels_times = torch.tensor(nodes_labels_times)

		return nodes_labels_times


	def load_transactions(self, elliptic_args, tar_archive):
		try:
			data = u.load_data_from_tar(elliptic_args.edges_file, tar_archive, type_fn=float, tensor_const=torch.LongTensor)
		except Exception as e:
			print('== Exception: ',e)
			traceback.print_exc()
			filepath = os.path.join(elliptic_args.folder, elliptic_args.edges_file)
			data = u.load_data_from_file(filepath, type_fn=float, tensor_const=torch.LongTensor)
		tcols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})

		data = torch.cat([data,data[:,[1,0,2]]])

		self.max_time = data[:,tcols.time].max()
		self.min_time = data[:,tcols.time].min()

		return {'idx': data, 'vals': torch.ones(data.size(0))}
