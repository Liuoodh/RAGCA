import torch
import tqdm
from torch import nn
from torch import optim

from models import KBCModel
from regularizers import Regularizer

from datasets import Sampler,Dataset


# os.environ['CUDA_VISIBLE_DEVICES'] = device




class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer,train_mode,context_weight,dataset: Dataset, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_mode = train_mode
        self.dataset = dataset
        self.examples = torch.from_numpy(dataset.get_train().astype('int64'))
        self.context_weight = context_weight

    def epoch(self,n_node:int):

        sampler = Sampler(data=self.examples,n_ent=n_node)
        if self.train_mode=='multivariate':
            loss = nn.CrossEntropyLoss(reduction='mean')
            # loss = nn.MultiMarginLoss()
        else:
            loss = nn.BCELoss()
        with tqdm.tqdm(total=self.examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            while not sampler.is_empty():
                if self.train_mode=='multivariate':
                    input_batch = sampler.batchify(self.batch_size,'cuda',self.train_mode)
                    predictions, factors ,predictions_context, factors_context= self.model.forward(input_batch)
                    truth = input_batch[:, 2]

                    # predictions =predictions/16

                    # predictions_context =predictions_context/16

                    l_fit = loss(predictions, truth)
                    l_fit_context = loss(predictions_context, truth)

                    l_reg = self.regularizer.forward(factors)

                    l_reg_context = self.regularizer.forward(factors_context)

                    l = (1-self.context_weight)*(l_fit+l_reg)+self.context_weight*(l_fit_context+l_reg_context)

                    batch_size = input_batch.shape[0]
                else:
                    batch= sampler.batchify(self.batch_size,'cuda',self.train_mode)
                    batch_size = batch.shape[0]
                    truth = self.dataset.get_bce_label(batch,n_node).to(torch.float)

                    predictions, factors = self.model.forward(batch)
                    l_fit = loss(predictions, truth)
                    l_reg = self.regularizer.forward(factors)
                    l = l_fit+l_reg
                # l = l_fit+l_reg

                # for name, param in self.model.named_parameters():
                #         print(f'Parameter: {name}, Gradient norm: {param.grad}')
                self.optimizer.zero_grad()

                # nn.utils.clip_grad_norm(self.model.parameters(),max_norm=2)
                l.backward()
                self.optimizer.step()
                # b_begin += self.batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=f'{l.item():.0f}')
            return l

