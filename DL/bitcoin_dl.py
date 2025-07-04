import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import utils as u
from collections import Counter
from torch_geometric.utils import stochastic_blockmodel_graph
from tqdm import tqdm
from .sbm_seed import stochastic_blockmodel_graph_with_seed

class bitcoin_dataset():
    def __init__(self,args, ood_mode=None):
        assert args.task in ['link_pred', 'edge_cls'], 'bitcoin only implements link_pred or edge_cls'
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.bitcoin_args = u.Namespace(args.bitcoin_args)

        #build edge data structure
        edges = self.load_edges(args.bitcoin_args)

        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.bitcoin_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals


        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2
        print("----- ID edges:",len(self.edges)) 
        print(f"self.edges['idx'].shape: {self.edges['idx'].shape}") # [47424, 4]
        # print(f"self.edges['idx'][:100]: {self.edges['idx'][:100]}")
        print(self.edges["idx"][:,2])
        print(torch.unique(self.edges["idx"][:,3]))
        print(f"self.edges['vals']: {self.edges['vals'][:100]}")
        print(f"self.max_time,self.min_time: {self.max_time},{self.min_time}")
        print("--------------:",len(self.edges)) 

        # import pandas as pd
        # df_idx = pd.DataFrame(self.edges['idx'].numpy(), columns=['idx1', 'idx2', 'idx3', 'idx4'])
        # df_idx['vals'] = self.edges['vals'].numpy()
        # output_file = f'./DL/data_SM/edges_{args.data}.csv'
        # df_idx.to_csv(output_file, index=False)

        if ood_mode == "SM":  
            out_degrees = Counter(self.edges['idx'][:, 0].numpy())  
            in_degrees = Counter(self.edges['idx'][:, 1].numpy())  
            degree_of_ones = sum(1 for value in out_degrees.values() if value == 1)
            print(f"Number of keys with value 1: {degree_of_ones}/{len(out_degrees)}")


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
                print(f"self.edges['idx'].shape: {self.edges['idx'].shape}")
                print(self.edges["idx"][:3])
                print(torch.unique(self.edges["vals"]))
            else:
                for timestep in tqdm(range(self.min_time, int(self.max_time) + 1), desc=f"Generating SM Graphs for {args.data}"):
                    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)    
                    cols_time = torch.tensor([timestep] * edge_index.size(1))
                    cols_rating = torch.randint(0, 2, (edge_index.size(1),), dtype=torch.int64) # rating
                    edge_index_time = torch.cat([edge_index.T, cols_time.unsqueeze(1), cols_rating.unsqueeze(1)],  dim=1)
                    SM_edges.append(edge_index_time)

                self.edges["idx"] = torch.cat(SM_edges, dim=0)
                print(self.edges["idx"][:3])
                self.edges["vals"] = torch.randint(1, 3, (self.edges["idx"].size(0),), dtype=torch.int64)
                print(self.edges["vals"])
                torch.save(self.edges, save_path)
            # g_edges = self.edges['idx'][self.edges['idx'][:,2] == 10]
            # g_edges = g_edges[:,:2]
            # print(g_edges)
            # for timestep in range(self.min_time,int(self.max_time)+1):    




        elif ood_mode == "FI":
            print("BITCOIN FI data will be got in tasker")


    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings

    def prepare_node_feats(self,node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self,edges):
        idx = edges[:,[self.ecols.FromNodeId,
                       self.ecols.ToNodeId,
                       self.ecols.TimeStep]]

        vals = edges[:,self.ecols.Weight]
        return {'idx': idx,
                'vals': vals}

    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_edges(self,bitcoin_args):
        file = os.path.join(bitcoin_args.folder,bitcoin_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges

    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
