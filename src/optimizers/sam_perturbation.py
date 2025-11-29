import torch

class SAMPerturbation:
    def __init__(self, model, rho=0.05):
        self.model = model
        self.rho = rho
        self.backup = {}

    def perturb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                norm = torch.norm(grad)
                if norm != 0:
                    e_w = (self.rho / norm) * grad
                    param.data.add_(e_w)
                    self.backup[name] = e_w

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.sub_(self.backup[name])
        self.backup = {}
