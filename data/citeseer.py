import time
import torch
import dgl.data

class CiteseerDGL(torch.utils.data.Dataset):

    def __init__(self, stage = 'train'):
        self.stage = stage
        self.graph = dgl.data.CiteseerGraphDataset()[0]
        self.graph.ndata['train_mask'] = torch.ones(self.graph.num_nodes(), dtype=torch.bool)
        self.graph.ndata['train_mask'][self.graph.ndata['val_mask']] = False
        self.graph.ndata['train_mask'][self.graph.ndata['test_mask']] = False
        self.graph.edata['feat'] = torch.ones([self.graph.num_edges(), 1]).float()
        print(f"new num of train is {torch.count_nonzero(self.graph.ndata['train_mask'])}")
    
    def __getitem__(self, i):
        #assert i == 0
        return self.graph, self.graph.ndata['label']

    def __len__(self):
        if self.stage == 'train':
            return 200
        else:
            return 1

class CiteseerDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading Citeseer Dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name  = name
        self.train = CiteseerDGL("train")
        self.val   = CiteseerDGL("valid")
        self.test  = CiteseerDGL("test")

        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        return samples[0]

if __name__ == '__main__':
    dataset = CiteseerDataset('Citeseer')
    print(dataset.train.__getitem__(0))