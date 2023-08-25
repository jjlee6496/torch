import torch.optim as optim

def get_optimzer(model, optimizer_type ,learning_rate, momentum=None):
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        if momentum is None:
            raise ValueError('Momentum must be provided for SGD optimizer')
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError('Unsupported optimizer type.')