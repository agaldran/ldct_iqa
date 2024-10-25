import os, json, sys, time, random, os.path as osp
import numpy as np
import torch
from tqdm import trange, tqdm
from utils.data_handling import get_class_loaders
from utils.evaluation import evaluate_cls
from utils.get_model import get_arch
from utils.sam import SAM
from scipy.stats import pearsonr, spearmanr, kendalltau

def set_seed(seed_value, use_cuda):
    if seed_value is not None:
        np.random.seed(seed_value)  # cpu vars
        torch.manual_seed(seed_value)  # cpu  vars
        random.seed(seed_value)  # Python
        # torch.use_deterministic_algorithms(True)

        if use_cuda:
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True

def save_model(path, model):
    os.makedirs(path, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()},
                 osp.join(path, 'model_checkpoint.pth'))

def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_args_parser():
    import argparse

    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    parser = argparse.ArgumentParser(description='Training for Biomedical Image Classification')
    parser.add_argument('--csv_train', type=str, default='data/train_f1.csv', help='path to training data csv')
    parser.add_argument('--data_path', type=str, default='data/LDCTIQAG2023_train/image', help='path to training images')
    parser.add_argument('--model', type=str, default='resnet18', help='architecture')
    parser.add_argument('--n_heads', type=int, default=1, help='if greater than 1, use NHeads Ensemble Learning')
    parser.add_argument('--balanced_mh', type=str2bool, nargs='?', const=True, default=False, help='Balance loss on heads')
    parser.add_argument('--random_heads', type=str2bool, nargs='?', const=True, default=True, help='Random head-class distribution')
    parser.add_argument('--overall_loss', type=str, default='ce', help='overall loss on top of head losses')
    parser.add_argument('--hypar', type=float, default=-1, help='some overall losses have hyper-parameter, set -1 for their defaults')
    parser.add_argument('--cycle_lens', type=str, default='10/5', help='cycling config (nr cycles/cycle len)')

    parser.add_argument('--opt', default='nadam', type=str, choices=('sgd', 'adamw', 'nadam'), help='optimizer to use (sgd | adamW | nadam)')
    parser.add_argument('--sam', type=str2bool, nargs='?', const=True, default=False, help='use sam wrapping optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')  # adam -> 1e-4, 3e-4
    parser.add_argument('--momentum', default=0., type=float, help='sgd momentum')
    parser.add_argument('--epsilon', default=1e-8, type=float, help='adamW epsilon for numerical stability')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='label smoothing (default: 0.0)')

    parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (auc)')
    parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512')
    parser.add_argument('--save_path', type=str, default=None, help='path to save model (defaults to None => debug mode)')
    parser.add_argument('--num_workers', default=16, type=int, help='number of data loading workers (default: 6)')
    parser.add_argument('--device', default='cuda', type=str, help='device (cuda or cpu, default: cuda)')
    parser.add_argument('--seed', type=int, default=None, help='fixes random seed (slower!)')

    # parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=True, help='avoid saving anything')

    args = parser.parse_args()

    return args

def disable_bn(model):
    for module in model.modules():
      if isinstance(module, torch.nn.BatchNorm3d):
        module.eval()
def enable_bn(model):
    model.train()

def run_one_epoch(model, optimizer, ce_weights, overall_criterion, loader, scheduler=None, assess=False):

    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train: model.train()
    else: model.eval()
    ce = torch.nn.functional.cross_entropy
    probs_class_all, preds_class_all, labels_all = [], [], []
    run_loss_class = 0
    weights = [torch.tensor(w, dtype=torch.float32).to(device) for w in ce_weights]

    with trange(len(loader)) as t:
        n_elems, running_loss_class = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            logits = model(inputs)

            if model.n_heads == 1:
                loss_class = ce(logits, labels)
            else:
                if train:
                    losses = [ce(logits[:, i, :], labels, weights[i]) for i in range(model.n_heads)]
                    overall_loss = overall_criterion(logits.mean(dim=1), labels)
                    losses.append(overall_loss)
                    loss_class = torch.mean(torch.stack(losses))
                else:
                    loss_class = ce(logits, labels)

            if train:  # only in training mode
                loss_class.backward()
                if isinstance(optimizer, SAM):
                    optimizer.first_step(zero_grad=True)
                    logits = model(inputs)
                    if model.n_heads == 1:
                        loss_class = ce(logits, labels)
                    else:
                        losses = [ce(logits[:, i, :], labels, weights[i]) for i in range(model.n_heads)]
                        overall_loss = ce(logits.mean(dim=1), labels)
                        losses.append(overall_loss)
                        loss_class = torch.mean(torch.stack(losses))
                    # compute BN statistics only in the first backwards pass
                    disable_bn(model)
                    loss_class.backward()
                    enable_bn(model)
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                lr = get_lr(optimizer)
                scheduler.step()
                optimizer.zero_grad()
            if assess:
                probs_class = logits.softmax(dim=1)

                preds_class = np.argmax(probs_class.detach().cpu().numpy(), axis=1)
                probs_class_all.extend(probs_class.detach().cpu().numpy())
                preds_class_all.extend(preds_class)
                labels_all.extend(labels.cpu().numpy())
            # Compute running loss
            running_loss_class += loss_class.detach().item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss_class = running_loss_class / n_elems
            if train: t.set_postfix(loss_lr="{:.4f}/{:.6f}".format(run_loss_class, lr))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss_class)))
            t.update()
    if train: print('Class Loss= {:.4f} -- LR = {:.7f}'.format(loss_class, lr))
    if assess: return np.stack(preds_class_all), np.stack(probs_class_all), np.stack(labels_all), run_loss_class
    return None, None, None, None

def train_one_cycle(model, optimizer, ce_weights, overall_criterion, train_loader, scheduler, cycle=0):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]

    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle+1, epoch+1, cycle_len))
        if epoch == cycle_len-1: assess=True # only get probs/preds/labels on last cycle
        else: assess = False
        tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch(model, optimizer, ce_weights, overall_criterion, train_loader, scheduler, assess)

    return tr_preds, tr_probs, tr_labels, tr_loss

def train_model(model, optimizer, ce_weights, overall_criterion, train_loader, val_loader, scheduler, metric, save_path):

    n_cycles = len(scheduler.cycle_lens)
    best_loss, best_auc, best_mcc, best_cycle = 10, 0, 0, 0
    best_plcc, best_srocc, best_krocc = 0, 0, 0
    all_tr_aucs, all_vl_aucs, all_tr_mccs, all_vl_mccs = [], [], [], []
    all_tr_losses, all_vl_losses = [], []
    all_tr_plccs, all_vl_plccs, all_tr_sroccs = [], [], []
    all_vl_sroccs, all_tr_kroccs, all_vl_kroccs = [], [], []

    n_classes = len(train_loader.dataset.classes)
    class_names = ['C{}'.format(i) for i in range(n_classes)]
    print_conf, text_file_train, text_file_val = False, None, None

    for cycle in range(n_cycles):
        print('\nCycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # train one cycle
        _, _, _, _ = train_one_cycle(model, optimizer, ce_weights, overall_criterion, train_loader, scheduler, cycle=cycle)

        with torch.inference_mode():
            tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch(model, None, np.ones_like(ce_weights), overall_criterion, train_loader, assess=True)
            vl_preds, vl_probs, vl_labels, vl_loss = run_one_epoch(model, None, np.ones_like(ce_weights), overall_criterion, val_loader, assess=True)

        if save_path is not None:
            print_conf = True
            text_file_train = osp.join(save_path,'performance_cycle_{}.txt'.format(str(cycle+1).zfill(2)))
            text_file_val = osp.join(save_path, 'performance_cycle_{}.txt'.format(str(cycle+1).zfill(2)))


        tr_auc, tr_mcc, tr_acc, tr_auc_all = evaluate_cls(tr_labels, tr_preds, tr_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_train, loss=tr_loss)
        vl_auc, vl_mcc, vl_acc, vl_auc_all = evaluate_cls(vl_labels, vl_preds, vl_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_val, loss=vl_loss)

        print('Train||Val Loss: {:.4f}||{:.4f} - AUC: {:.2f}||{:.2f} - MCC: {:.2f}||{:.2f}'.format(tr_loss, vl_loss,
                                                                                                   100 * tr_auc, 100 * vl_auc,
                                                                                                   100 * tr_mcc, 100 * vl_mcc))
        tr_plcc, vl_plcc = abs(pearsonr(tr_preds, tr_labels)[0]), abs(pearsonr(vl_preds, vl_labels)[0])
        tr_srocc, vl_srocc = abs(spearmanr(tr_preds, tr_labels)[0]), abs(spearmanr(vl_preds, vl_labels)[0])
        tr_krocc, vl_krocc = abs(kendalltau(tr_preds, tr_labels)[0]), abs(kendalltau(vl_preds, vl_labels)[0])

        print('Train||Val Pearson: {:.4f}||{:.4f} - Spearman: {:.4f}||{:.4f} - Tau: {:.4f}||{:.4f}'.format(tr_plcc, vl_plcc, tr_srocc,
                                                                                                           vl_srocc, tr_krocc, vl_krocc))

        all_tr_aucs.append(tr_auc)
        all_vl_aucs.append(vl_auc)
        all_tr_mccs.append(tr_mcc)
        all_vl_mccs.append(vl_mcc)
        all_tr_plccs.append(tr_plcc)
        all_vl_plccs.append(vl_plcc)
        all_tr_sroccs.append(tr_srocc)
        all_vl_sroccs.append(vl_srocc)
        all_tr_kroccs.append(tr_krocc)
        all_vl_kroccs.append(vl_krocc)
        all_tr_losses.append(tr_loss)
        all_vl_losses.append(vl_loss)

        # check if performance was better than anyone before and checkpoint if so
        if vl_auc > best_auc:
            print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, 100*best_auc, 100*vl_auc))
            best_loss, best_auc, best_cycle = vl_loss, vl_auc, cycle+1
            best_auc = vl_auc
            best_mcc = vl_mcc
            best_plcc = vl_plcc
            best_srocc = vl_srocc
            best_krocc = vl_krocc
            if save_path is not None: save_model(save_path, model)
        else:
            print('-------- Best AUC so far {:.2f} at cycle {:d} --------'.format(100 * best_auc, best_cycle))


    del model
    torch.cuda.empty_cache()
    return best_auc, best_mcc, best_loss, best_plcc, best_srocc, best_krocc, all_tr_aucs, all_vl_aucs, all_tr_mccs, \
           all_vl_mccs, all_tr_plccs, all_vl_plccs, all_tr_sroccs, all_vl_sroccs, all_tr_kroccs, all_vl_kroccs, \
           all_tr_losses, all_vl_losses, best_cycle

def main(args):
    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    # reproducibility
    set_seed(args.seed, use_cuda)

    save_path = args.save_path
    if save_path is not None:
        save_path=osp.join('experiments', save_path)
        args.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        config_file_path = osp.join(save_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Prepare training data
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    train_loader, val_loader = get_class_loaders(csv_train, csv_val, args.data_path, tg_size, args.batch_size, args.num_workers)
    num_classes = len(train_loader.dataset.classes)
    n_heads = args.n_heads
    if args.model=='bit_resnext50_1':
        train_loader.dataset.normalize.mean, train_loader.dataset.normalize.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        val_loader.dataset.normalize.mean, val_loader.dataset.normalize.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    # Prepare model for training
    model = get_arch(args.model, num_classes, n_heads)

    model.to(device)

    # Prepare optimizer and scheduler
    weight_decay = 0
    if weight_decay > 0:
        # it's okay to use weight decay, but do not apply to it normalization layers
        # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    # Prepare optimizer and scheduler
    if args.opt == 'adamw':
        if args.sam:
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'nadam':
        if args.sam:
            base_optimizer = torch.optim.NAdam
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        if args.sam:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise RuntimeError('Invalid optimizer {}. Only SGD and AdamW are supported.'.format(args.opt))


    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
    cycle_lens = list(map(int, cycle_lens))
    if len(cycle_lens) > 2:
        sys.exit('cycles should be specified as a pair n_cycles/cycle_len')
    cycle_lens = cycle_lens[0] * [cycle_lens[1]]

    if args.sam:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=cycle_lens[0]*len(train_loader), eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_lens[0]*len(train_loader), eta_min=0)
    setattr(optimizer, 'max_lr', args.lr)  # store maximum lr inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)


    if args.overall_loss == 'ce':
        overall_criterion = torch.nn.CrossEntropyLoss()
    elif args.overall_loss == 'ce+rps': # adaptive version of FL has no hyper-parameter
        from utils.rps_loss import RPS
        overall_criterion = RPS(alpha_ce=1, beta_rps=args.hypar)
    elif args.overall_loss == 'rps': # adaptive version of FL has no hyper-parameter
        from utils.rps_loss import RPS
        overall_criterion = RPS(alpha_ce=0, beta_rps=args.hypar)


    classes = np.arange(num_classes)
    if args.random_heads:
        random.shuffle(classes)
    more_weighted_classes_per_head = np.array_split(classes, n_heads)
    ce_weights = []
    for c in more_weighted_classes_per_head:
        # weights for this head are 2 for classes in c and 1/2 for classes that are not in c
        w = [float(n_heads) if i in c else 1/n_heads for i in np.arange(num_classes)]
        if args.balanced_mh:
            # do not use weighted-CE, so this is an Unperturbed Bycephal
            w = num_classes*[1]
        ce_weights.append(w)
    print('Using class weights: ', ce_weights)

    # Start training
    start = time.time()
    best_auc, best_mcc, best_loss, best_plcc, best_srocc, best_krocc, all_tr_aucs, all_vl_aucs, all_tr_mccs, \
    all_vl_mccs, all_tr_plccs, all_vl_plccs, all_tr_sroccs, all_vl_sroccs, all_tr_kroccs, all_vl_kroccs, \
    all_tr_losses, all_vl_losses, best_cycle = train_model(model, optimizer, ce_weights, overall_criterion, train_loader, val_loader, scheduler, metric, save_path)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('done')

    if save_path is not None:
        with open(osp.join(save_path, 'log.txt'), 'w') as f:
            print('Best AUC = {:.2f}\nBest MCC = {:.2f}\nBest loss = {:.2f}'.format(100*best_auc, 100*best_mcc, best_loss), file=f)
            print('Best Pearson = {:.4f}\nBest Spearman = {:.4f}\nBest Kendall = {:.4f}\nBest cycle = {}\n'.format(best_plcc, best_srocc, best_krocc, best_cycle), file=f)

            for j in range(len(all_tr_aucs)):
                print('Cycle = {} -> AUC={:.2f}/{:.2f}, MCC={:.2f}/{:.2f}, Loss={:.4f}/{:.4f}'.format(j+1,
                            100*np.mean(all_tr_aucs[j]), 100*np.mean(all_vl_aucs[j]),
                            100*all_tr_mccs[j], 100*all_vl_mccs[j], all_tr_losses[j], all_vl_losses[j]), file=f)
                print('Cycle = {} -> Pearson={:.4f}/{:.4f}, Spearman={:.4f}/{:.4f}, Kendall={:.4f}/{:.4f}'.format(j+1,
                            all_tr_plccs[j], all_vl_plccs[j], all_tr_sroccs[j], all_vl_sroccs[j], all_tr_kroccs[j], all_vl_kroccs[j]), file=f)
                print(82*'-', file=f)

            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))
if __name__ == "__main__":
    args = get_args_parser()
    main(args)
