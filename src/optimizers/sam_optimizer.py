import torch

class SAMOptimizer:
    def __init__(self, model, base_optimizer, rho=0.05, **kwargs):
        self.model = model
        self.base_optimizer = base_optimizer(model.parameters(), **kwargs)
        self.rho = rho
        self.backup = {}  

    def first_step(self, zero_grad=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = torch.norm(param.grad)
                if grad_norm != 0:
                    e_w = (self.rho / grad_norm) * param.grad
                    param.data.add_(e_w)
                    self.backup[name] = e_w.clone()  

        if zero_grad:
            self.model.zero_grad()

    def second_step(self, zero_grad=True):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.sub_(self.backup[name])  
        self.backup = {}

        self.base_optimizer.step()

        if zero_grad:
            self.model.zero_grad()

    def zero_grad(self):
        self.model.zero_grad()
