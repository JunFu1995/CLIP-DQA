from cgi import test
import os
from scipy import stats
import random
import numpy as np
from datasets.dataloader_gl import *
import torch.nn as nn
import sys
import yaml
import models 
import torch.optim.lr_scheduler as LS
import utils.utils as utils
class Manager(object):
    def __init__(self, options, path, percentage, rd):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        self.percentage = percentage
        self.round = rd

        # Network.
        self._net,_ = models.buildModel(options['model'], 'cfg_8')

        # Criterion.
        self._criterion = torch.nn.MSELoss().cuda()

        # Solver.
        # , {'params':self._net.image_encoder.parameters(),'lr':1e-5,'weight_decay':0.01}
        self._solver = torch.optim.Adam([{'params':self._net.prompt_learner.parameters()}], self._options['base_lr']) 

        dn = self._options['dataset']
        self._train_loader = DataLoader(dn, self._path[dn], self._options['train_index'], \
                                        batch_size=self._options['batch_size'], istrain=True, patch_num=1).get_data()

        self._test_loader = DataLoader(dn, self._path[dn], self._options['test_index'], \
                                       istrain=False).get_data()

        self.ps = 224
        self.unfold = nn.Unfold(kernel_size=(self.ps,self.ps), stride=64) # 224
        self.savePath = os.path.join(options['savePath'], '%s_%s_%d_best.pth'%(options['model'], options['dataset'], options['n_ctx']))
        self.testDataPath = os.path.join(options['savePath'], '%s_%s_best'%(options['model'], options['dataset']))
        self.scheduler = LS.MultiStepLR(self._solver, milestones=[20, 40, 190, 120, 200, 300], gamma=0.5) #0.47)#10
        self.scheduler.last_epoch = 0 #args.start_epoch
    def train(self):
        """Train the network."""
        print('Training.')
        best_srcc = 0.0
        best_plcc = 0.0
        best_krcc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self._options['epochs']):
            epoch_loss = []
            pscores = []
            tscores = []
            for X, y, path, z in self._train_loader:
                # Data.
                X = X.cuda()  # local info
                y = y.cuda().float() # label
                z = z.cuda()# global info
                # Forward pass.
                score = self._net(X,z)
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Backward pass.
                loss = self._criterion(score, y.view(len(score), 1).detach())
                epoch_loss.append(loss.item())
                # Prediction.
                pscores = pscores + score.cpu().tolist()
                tscores = tscores + y.cpu().tolist()
                loss.backward()
                self._solver.step()
            train_srcc, _ = stats.spearmanr(pscores, tscores)
            #self.scheduler.step()
            with torch.no_grad():
                test_srcc, test_plcc, test_data, test_krcc = self._consitency(self._test_loader)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                best_krcc = test_krcc
                best_epoch = t + 1
                # save model 
                torch.save(self._net.prompt_learner.state_dict(), self.savePath)
                np.save(self.testDataPath, test_data)
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t%4.4f\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_krcc))

        print('Best at epoch %d, test srcc %.4f, test plcc %.4f' % (best_epoch, best_srcc, best_plcc))
        return best_srcc, best_plcc, best_krcc

    def _consitency(self, data_loader):
        self._net.train(False)
        num_total = 0
        pscores = []
        tscores = []
        batch_size = 128
        test_data = {}
        for X, y, path, z in data_loader:
            # Data.
            X = X.cuda()  # local info
            y = y.cuda().float() # label
            z = z.cuda()# global info

            X_sub = self.unfold(X).view(1, X.shape[1], self.ps, self.ps, -1)[0]
            X_sub = X_sub.permute(3,0,1,2)

            img = torch.split(X_sub, batch_size, dim=0)
            pred_s = []
            for i in img:
                pred = self._net(i, z.repeat(i.shape[0],1,1,1)) #.repeat(i.shape[0],1)
                pred_s += pred.detach().cpu().tolist()
            score = np.mean(pred_s)
            pscores = pscores + [score]
            tscores = tscores + y.cpu().tolist()
            test_data[path] = [score, y.cpu().tolist()[0]]
            num_total += y.size(0)
        test_srcc, _ = stats.spearmanr(pscores, tscores)
        #fscores     = utils.logistic5_regression(pscores, tscores)
        test_plcc, _ = stats.pearsonr(pscores, tscores)
        test_krcc, _ = stats.kendalltau(pscores, tscores)
        self._net.train(True)  # Set the model to training phase
        return test_srcc, test_plcc, test_data, test_krcc

class flushfile:
    #https: // stackoverflow.com / questions / 230751 / how - can - i - flush - the - output - of - the - print - function
  def __init__(self, f):
    self.f = f
  def write(self, x):
    self.f.write(x)
    self.f.flush()

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='test clip for iqa tasks.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-4,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=64, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=25, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset', dest='dataset', type=str, default='DHD',
                        help='dataset: live|csiq|tid2013|livec|mlive')
    parser.add_argument('--model', dest='model', type=str, default='CLIP-DQA',
                        help='dataset: live|csiq|tid2013|livec|mlive')   
    parser.add_argument('--n_ctx', dest='n_ctx', type=int, default=8,
                        help='dataset: live|csiq|tid2013|livec|mlive')   
    parser.add_argument('--gpuid', type=str, default='0', help='GPU ID')
    parser.add_argument('--percentage', type=float, default=0.8, help='training portion')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print(args.percentage)

    f = open('./log/%s_%s.log'%(args.model, args.dataset), 'w')
    sys.stdout = flushfile(f)

    seed = 10  # random.randint(1, 10000)
    print("Random Seed: ", seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs)
    # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    if args.dataset == 'IVCDD':
        args.epochs = 350

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset': args.dataset,
        'fc': [],
        'train_index': [],
        'test_index': [],
        'model': args.model,
        'n_ctx': args.n_ctx
    }

    path = {
        'DHD': '/home/fujun/datasets/iqa/DHD',
        'exBeDDE': '/home/fujun/datasets/iqa/exBeDDE',
    }

    if options['dataset'] == 'DHD':
        index = list(range(0, 250))
    elif options['dataset'] == 'exBeDDE':
        index = list(range(0, 12))
 
    roudNum = 10
    srcc_all = np.zeros((1, roudNum), dtype=np.float64)
    plcc_all = np.zeros((1, roudNum), dtype=np.float64)
    krcc_all = np.zeros((1, roudNum), dtype=np.float64)
    for i in range(0, roudNum): # 5 for idea fast check
        print("====================round %d=====================" % i )
        # randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(args.percentage * len(index))]
        test_index = index[round(args.percentage * len(index)):len(index)]

        savePath = os.path.join('./save/', args.model, 'round%d' % i)
        if not os.path.isdir(savePath):
            os.makedirs(savePath)

        options['train_index'] = train_index
        options['test_index'] = test_index
        options['savePath'] = savePath
        # train the fully connected layer only
        options['fc'] = True
        options['base_lr'] = 1e-4
        manager = Manager(options, path, args.percentage, i)
        best_srcc, best_plcc, best_krcc = manager.train()
        srcc_all[0][i] = best_srcc
        plcc_all[0][i] = best_plcc
        krcc_all[0][i] = best_krcc

    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)
    krcc_mean = np.mean(krcc_all)
    
    print('srcc', srcc_all)
    print('plcc', plcc_all)
    print('krcc', krcc_all)

    print('average mean srcc:%4.4f, plcc:%4.4f, krcc:%4.4f' % (srcc_mean, plcc_mean, krcc_mean))
    print('average std srcc:%4.4f, plcc:%4.4f, krcc:%4.4f' % (srcc_all.std(), plcc_all.std(), krcc_all.std()))

if __name__ == '__main__':
    main()
