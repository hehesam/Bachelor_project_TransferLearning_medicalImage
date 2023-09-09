from torch import nn

def truncate_densenet(net: nn.Module, trunc: int):
    if trunc==-1:
        return net
    else:
        # print(list(net..children()))
        # asdfg
        return nn.Sequential(*list(net.children())[:4+trunc*2])
    

def densenet121(pretrained : str ="imagenet", trunc : int=-1, classes : int=7):
    # sainity check
    assert pretrained in SUPPORTED_WEIGHTS, "weights not found"
    
    if pretrained=="imagenet":
        print("load imagenet pretrained weights")
        net = models.densenet121(pretrained=True).features
    elif pretrained=="CheXpert":
        print("load chexpert pretrained weights")
        net = torch.load("../checkpoints/CheXpert/densenet121_random_-1_1/best.pt").module.to(torch.device("cpu"))
    elif pretrained=="random":
        net = models.densenet121(pretrained=False).features
    else:
        sys.exit("wrong specification")
        
    net = truncate_densenet(net, trunc)
    test_input = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(net(test_input).shape)
    fc = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                          out_features=classes,
                          bias=True)
            ])
    net = nn.Sequential(net, fc)
        
    return net


