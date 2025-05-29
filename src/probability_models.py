import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self,
                nfeat_in=3,
                nfeat_out=28,
                nhidden=256,
                nlayers=3,
                dropout=None
                ):
        super().__init__()
        self.relu = nn.ReLU()
        self.nfeat_in = nfeat_in
        self.nfeat_out = nfeat_out
        self.nlayers = nlayers
        self.nhidden = nhidden

        if dropout != None:
            assert len(dropout) == nlayers
            drlist = []
            for dr in dropout:
                if dr !=None :
                    drlist.append(nn.Dropout(p=dr))
                else:
                    drlist.append(nn.Identity())
        else:
            drlist = [nn.Identity()  for _ in range(nlayers)]

        self.fflayers = []
        self.fflayers.append(nn.Linear(nfeat_in, nhidden))
        self.fflayers.append(drlist.pop(0))
        for _ in range(1,nlayers-1):
            self.fflayers.append(self.relu)
            self.fflayers.append(nn.Linear(nhidden, nhidden))
            self.fflayers.append(drlist.pop(0))

        self.fflayers.append(self.relu)
        
        self.classifier_head = nn.Linear(nhidden, nfeat_out)
        self.fflayers = nn.Sequential(*self.fflayers)

    def forward(self, x):
        x = self.fflayers(x)
        x = self.classifier_head(x)
        return x
