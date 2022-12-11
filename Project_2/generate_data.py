import torch 

def check_circle(input):
    target = torch.zeros(input.size(0))
    target += ((torch.pow(input[:, 0] - .5, 2) + torch.pow(input[:, 1] - .5, 2)) <= torch.div(1, torch.sqrt(torch.tensor([2 * torch.pi]))))  # avoiding python loop
    return target

def generate_disc(nb, normalization):
    input = torch.distributions.uniform.Uniform(0, 1).sample([nb,2]) 
    target = check_circle(input)
    
    if normalization:
        input -= input.mean()
        input /= input.std()
    return input, target

def generate_disc_set(nb , device, split= .7, normalization=False):
    input, target = generate_disc(nb, normalization)
    input, target = input.to(device), target.to(device)
    
    if split < 1:
        train_size = int(split * target.size(0))
    else:
        raise Exception("Invalid value for split")
    
    train_input = input[:train_size]
    test_input = input[train_size:]
    train_target = target[:train_size]
    test_target = target[train_size:]
    return train_input, test_input, train_target, test_target