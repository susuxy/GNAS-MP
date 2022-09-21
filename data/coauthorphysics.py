import time
import torch
import dgl.data

class CoauthorPhysicsDGL(torch.utils.data.Dataset):

    def __init__(self, stage = "train"):
        self.stage = stage
        self.graph = dgl.data.CoauthorPhysicsDataset()[0]
        print(self.graph)
        self.graph.edata['feat'] = torch.ones([self.graph.num_edges(), 1]).float()
    
    def __getitem__(self, i):
        #assert i == 0
        return self.graph, self.graph.ndata['label']

    def __len__(self):
        if self.stage == 'train':
            return 400
        else:
            return 1

class CoauthorPhysicsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading CoauthorPhysics Dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name  = name
        self.train = CoauthorPhysicsDGL('train')
        self.val   = CoauthorPhysicsDGL('val')
        self.test  = CoauthorPhysicsDGL('test')

        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        return samples[0]

if __name__ == '__main__':
    dataset = CoauthorPhysicsDataset('CoauthorPhysics')
    print(dataset.train.__getitem__(0))