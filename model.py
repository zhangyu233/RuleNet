import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from torch.utils.data import DataLoader
from data import TrainDataset, TestDataset

class EmbeddingLayer(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma):
        super(EmbeddingLayer, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim , self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )


    def forward(self, sample):
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=head_part[:, 0]
        ).unsqueeze(1) #(b, 1, 500)

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=head_part[:, 1]
        ).unsqueeze(1) #(b, 1, 500, 500)

        nega_tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=tail_part.view(-1)
        ).view(batch_size, negative_sample_size, -1) #(b, 128, 500)

        posi_tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=head_part[:, 2]
        ).unsqueeze(1)

        t = torch.einsum('bad, badp-> bap', head, relation)
        nega_score = torch.einsum('bad, bkd-> bak', t, nega_tail).squeeze() #shape (b, negadim)
        posi_score = torch.einsum('bad, bkd-> bak', t, posi_tail).squeeze() #shape(b, 1)
        nega_score1 = F.logsigmoid(-nega_score).mean(dim = 1)
        posi_score1= F.logsigmoid(posi_score)
        # if args.uni_weight:
        # positive_sample_loss = - posi_score1.mean()
        # negative_sample_loss = - nega_score1.mean()
  

        return posi_score1, nega_score1, nega_score


    def get_embedding(self):
        return self.entity_embedding, self.relation_embedding

class Rule_Net(nn.Module):
    def __init__(self, num_layers, nentity, nrelation, hidden_dim, gamma ):
        super(Rule_Net, self).__init__()
        self.layers = num_layers
        self.EmbeddingLayer = EmbeddingLayer('1', nentity, nrelation, hidden_dim, gamma)
        # # self.W = nn.ModuleList()
        # for i in range (self.layers):
        #     self.W.extend(nn.Parameter(torch.Tensor(hidden_dim, 1)))
        self.W = [nn.Parameter(torch.Tensor(hidden_dim, 1)) for i in range(self.layers)] # initial
        for w in self.W:
            nn.init.uniform_(tensor=w, a=0, b=1)
        self.W =  nn.ParameterList(self.W)
        self.W2 = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.uniform_(tensor=self.W2, a=0, b=1)

    def forward(self, sample):
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
        head = torch.index_select(
            self.EmbeddingLayer.entity_embedding,
            dim=0,
            index=head_part[:, 0]
        ) #(b, 500)

        relations = self.EmbeddingLayer.relation_embedding #(r, 500, 500)
        posi_tail = torch.index_select(
            self.EmbeddingLayer.entity_embedding,
            dim=0,
            index=head_part[:, 2]
        ).unsqueeze(1)

        temp = list()
        r_atten = list()
        for l in range(self.layers):
            if l == 0:
                e1 = torch.einsum('bd,rdq->brq', head, relations)
                weight = torch.einsum('brq,qm->brm', e1, self.W[l])# (b,r,1)
                weight = F.softmax(weight, dim = 1)
                e1 = torch.einsum('brq,brt->bqt', e1, weight).squeeze() #（b,d）
                e1 = F.normalize(e1, p=2, dim=1)
                temp.append(e1)
                r_atten.append(weight)
            else:
                e1 = torch.einsum('bd,rdq->brq', temp[l-1], relations)
                weight = torch.einsum('brq,qm->brm', e1, self.W[l])# (b,r,1)
                weight = F.softmax(weight, dim = 1)
                e1 = torch.einsum('brq,brt->bqt', e1, weight).squeeze() #（b,d）
                e1 = F.normalize(e1, p=2, dim=1)
                temp.append(e1)
        final = torch.cat([torch.unsqueeze(r, 1) for r in temp], dim=1)
        # final = torch.cat(temp, dim = 1) #(b,l,d)
        weight2 = torch.einsum('bld,dk->blk', final, self.W2)
        weight2 = F.softmax(weight2, dim =1)
        final = torch.einsum('bld,blk->bd', final, weight2)
        final = F.normalize(final, p=2, dim=1) #
        match_score = torch.einsum('bd,bd->b', final, posi_tail.squeeze())
        match_score = F.logsigmoid(match_score)
        match_loss = -match_score.mean()
        return  r_atten, weight2, match_loss

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample,subsampling_weight = next(train_iterator)

        if args.cuda:
            device = torch.device(args.device)
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)


        posi_score1, nega_score1,  _  = model.EmbeddingLayer(( positive_sample, negative_sample))
        
        positive_sample_loss = - (subsampling_weight * posi_score1).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * nega_score1).sum() / subsampling_weight.sum()
        embed_loss = (positive_sample_loss + negative_sample_loss)/2

        r_atten, weight2, match_loss = model((positive_sample, negative_sample))
        regularization_log = {'regularization': 0}
        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.EmbeddingLayer.entity_embedding.norm(p=2) ** 2
            )
        # model.EmbeddingLayer.relation_embedding.norm(p=2).norm(p=2) ** 3
            embed_loss = embed_loss + args.regularization * regularization
            regularization_log = {'regularization': regularization.item()}

        if args.itertrain:
            embed_loss.backward()
            optimizer.step()
            # loss = embed_loss + match_loss
            loss = embed_loss
            loss.backward()
            optimizer.step()
        else:
            loss = embed_loss + match_loss
            loss.backward()
            optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'embed_loss': embed_loss.item(),
            'match_loss': match_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )
        logs = []
        step = 0
        total_steps = len(test_dataloader_tail)

        with torch.no_grad():
            for positive_sample, negative_sample, filter_bias in test_dataloader_tail:
                if args.cuda:
                    device = torch.device(args.device)
                    positive_sample = positive_sample.to(device)
                    negative_sample = negative_sample.to(device)
                    filter_bias = filter_bias.to(device)

                batch_size = positive_sample.size(0)

                _, _, score= model.EmbeddingLayer((positive_sample, negative_sample))
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)
                positive_arg = positive_sample[:, 2]


                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        return metrics
    



