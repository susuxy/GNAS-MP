import time
import torch
import dgl.data

class AmazonComputerDGL(torch.utils.data.Dataset):

    def __init__(self):
        self.graph = dgl.data.AmazonCoBuyComputerDataset()[0]
        self.graph.edata['feat'] = torch.ones([self.graph.num_edges(), 1]).float()
    
    def __getitem__(self, i):
        assert i == 0
        return self.graph, self.graph.ndata['label']

    def __len__(self):
        return 1

class AmazonComputerDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading AmazonComputer Dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name  = name
        base_graph = AmazonComputerDGL()
        self.train = base_graph
        self.val   = base_graph
        self.test  = base_graph

        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        return samples[0]

if __name__ == '__main__':
    dataset = AmazonComputerDataset('AmazonComputer')
    print(dataset.train.__getitem__(0))