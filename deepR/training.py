import torch

from tqdm.notebook import tqdm
from funcspec.tasks import get_continual_task, get_task_target
from models import *


"""
Deep Rewiring in Pytorch
Based on :
"Deep Rewiring: Training very sparse deep networks"
Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein
ICLR 2018
(https://arxiv.org/abs/1711.05136)
https://github.com/guillaumeBellec/deep_rewiring
"""

def get_device(model) : 
    try : 
        d = model.thetas[0].device
    except AttributeError : 
        d = model.fc1.weight.device
    return d

def train_step(batch, model, criterion, optimizer, params, global_rewire=True, soft_rewire=False, cost_fn=None) : 
    """ training function

    Args:
        batch (tuple): batch of data
        model (nn.Module): model to train
        criterion : loss function
        optimizer : optimiwer
        params (tuple): training parameters
        global_rewire (bool, optional): perform global rewiring. Defaults to True.
        soft_rewire (bool, optional): perform soft rewiring. Defaults to False.
        cost_fn (_type_, optional): use a connection cost. Defaults to None.

    Returns:
        tuple: number of new connections, and losses
    """
    #Forward
    model.train()
    
    data, target = batch
    device = get_device(model)
    data, target = data.to(device), target.to(device)
    output = model(data)

    reg_loss, alpha = torch.tensor(0.), 0.
    
    #Calculate Gradients
    loss = criterion(output, target)

    if cost_fn is not None : 
        alpha = 1.
        reg_loss = cost_fn[0](model.thetas)
        
    total_loss = loss #+ reg_loss*alpha
        
    optimizer.zero_grad()
    total_loss.backward()
    nb_connects = []
    try :
        for theta in model.thetas : 
            nb_connects.append(apply_grad(theta, params, soft=soft_rewire, cost_fn=cost_fn[1]))

        optimizer.step()
        
        #Rewire
        nb_reconnects = []
        if global_rewire : 
            nb_reconnections = sample_matrix_specific_reconnection_number_for_global_fixed_connectivity(model.thetas, model.sparsity_list)
            nb_reconnections = torch.stack(nb_reconnections)
            for th, nb_connect in zip(model.thetas, nb_reconnections) : 
                nb_reconnects.append(rewiring_global(th, nb_connect))

        elif soft_rewire : 
            for i, (th, th_clip) in enumerate(zip(model.thetas, model.th_clips)) : 
                rewiring_soft(th, th_clip)
                nb_reconnects.append((th>0).sum() - nb_connects[i])

        else : 
            for th, nb_connect in zip(model.thetas, model.nb_non_zero_list) : 
                nb_reconnects.append(rewiring(th, nb_connect))
    
    except AttributeError : 
        optimizer.step()
        nb_reconnects = [0]
    
            
    return nb_reconnects, loss.cpu().data.item(), reg_loss.cpu().data.item()
    
def train(model, train_loader, test_loader, n_epochs, criterion, optimizer, scheduler, params,
             task='none', cost_fn=None, global_rewire=True, soft_rewire=False, use_tqdm=True):
             
    train_losses = []
    test_accs, test_losses = [], []
    nb_reconnects = []

    descs = ['' for _ in range(2)]
    descs[0] = f'Epoch = 0, Training: '
    get_desc = lambda descs : descs[0] + descs[1]
    
    if use_tqdm : 
        pbar = tqdm(range(n_epochs), position=0, desc=get_desc(descs), leave=True)
    else :
        pbar = range(n_epochs)
        
    for epoch in pbar : 
        for batch_idx, (batch) in enumerate(train_loader) :
            if task == 'continual' : 
                data, target = batch
                current_task = np.round(batch_idx/len(train_loader))
                data, _ = get_continual_task(*batch, 'rotate', seed=current_task, n_tasks=2, n_classes=10) #rotate inputs every half epoch
                batch = data, target
            else : 
                current_task = task
            model.train()
            nb_reconnect, loss, reg_loss = train_step(batch, model, criterion, optimizer, params, global_rewire, soft_rewire, current_task, cost_fn)
            if soft_rewire :
                means = [f'{(t>0).float().mean().cpu().data.item()*100:.2f}' for t in model.thetas[:]]
                d = f'% Connects = {means}'
            else : 
                d = f'Nb Reconnect = {sum(nb_reconnect)}'

            descs[0] = f'Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({batch_idx/len(train_loader)*100:.0f}%)], Loss = {loss:.3f}, '
            if cost_fn is not None : descs[0] += f'Reg Loss = {reg_loss:.3f}, '
            descs[0] += f'{d} |'
            pbar.set_description(get_desc(descs))
            train_losses.append(loss), nb_reconnects.append(nb_reconnect)
        
        desc, test_loss, acc = test(model, criterion, test_loader, task, True)
        descs[1] = desc
        pbar.set_description(get_desc(descs))
        scheduler.step()
        test_losses.append(test_loss), test_accs.append(acc)
        if acc > 0.9 : 
            return train_losses, (test_losses, test_accs), nb_reconnects
        
    return train_losses, (test_losses, test_accs), nb_reconnects
        
def test(model, criterion, test_loader, task='none', verbose=True):
    model.eval()
    device = get_device(model)
    test_loss = 0
    correct = 0

    if task == 'continual' : 
        test_tasks = [0, 1]
    else :
        test_tasks = [task]

    for current_task in test_tasks :        
        with torch.no_grad():
            for batch in test_loader:

                if task == 'continual' : 
                    data, target = batch
                    data, _ = get_continual_task(*batch, 'rotate', seed=current_task, n_tasks=2, n_classes=10) #Target= 0: 0°, Target = 1 :180°
                    batch = data, target

                data, target = data.to(device), target.to(device)
                output = model(data)
                target, _ = get_task_target(target, current_task)
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)*len(test_tasks)
    acc = correct / (len(test_loader.dataset)*len(test_tasks))

    desc = str(' Test set: Average loss: {:.4f}, Accuracy: {:.0f}% ({}/{})'.format(
        test_loss, 100. * acc, correct, len(test_loader.dataset)*len(test_tasks)))
    
    return desc, test_loss, acc
    