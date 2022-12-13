import torch 

def check_circle(input):
    target = torch.zeros(input.size(0))
    target += ((torch.pow(input[:, 0] - .5, 2) + torch.pow(input[:, 1] - .5, 2)) <= 1/(torch.tensor(2) * torch.pi))  # avoiding python loop, put in the already squared radius to avoid unnecessary operations
    return target

def generate_disc(nb, normalization):
    input = torch.distributions.uniform.Uniform(0, 1).sample([nb,2]) 
    target = check_circle(input)
    
    if normalization:
        input -= input.mean()
        input /= input.std()
    return input, target

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def generate_disc_set(nb, split= .7, normalization=False, one_hot_labels=False):
    input, target = generate_disc(nb, normalization)
    input, target = input, target.unsqueeze(1)

    if one_hot_labels:  # in case we want two output neurons instead
        outside = (torch.zeros_like(target) == target)
        target = torch.cat((outside, target), 1)
    if split < 1:
        train_size = int(split * target.size(0))
    else:
        raise Exception("Invalid value for split")
    
    train_input = input[:train_size]
    test_input = input[train_size:]
    train_target = target[:train_size]
    test_target = target[train_size:]
    return train_input, train_target, test_input, test_target