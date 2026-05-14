import torch
import random
from einops import rearrange
import torchvision.transforms as torch_transforms

def horizontal_flip(p=0.5, single=False):
    #x: [B, X, Y, Z]
    def func(x):
        r = random.uniform(0, 1)
        dim = 0 if single else 1
        if r>p:
            x = torch.flip(x, dims=[dim])
        return x
    return func

def vertical_flip(p=0.5, single = False):
    #x: [B, X, Y, Z]
    def func(x):
        r = random.uniform(0, 1)
        dim = 1 if single else 2
        if r>p:
            x = torch.flip(x, dims=[dim])
        return x
    return func

def random_rotation(p=0.5, degrees=(-30, 30), single = False):
    rotation_transform = torch_transforms.RandomRotation(degrees=degrees, fill=17)
    def func(x):
        #x: [B, X, Y, Z]
        r = random.uniform(0, 1)
        if r>p:
            if single:
                x = rearrange(x, "x y z -> z x y")
                rotated_x = rotation_transform(x)
                x = rearrange(rotated_x, "z x y -> x y z")
            else:
                B = x.shape[0]
                x = rearrange(x, "b x y z -> (b z) x y").unsqueeze(1)
                rotated_x = rotation_transform(x)
                x = rearrange(rotated_x.squeeze(1), "(b z) x y -> b x y z", b=B)
        return x
    
    return func



def transforms(functions):
    def composed_function(input_value):
        result = input_value
        for func in functions:
            result = func(result)
        return result
    return composed_function



