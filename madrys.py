import torch
import torch.nn as nn
import torch.nn.functional as F


class MadrysLoss(nn.Module):


    def __init__(
        self,
        epsilon=8 / 255,
        step_size=None,
        perturb_steps=7,
        distance="L_inf",
        random_start=True,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.step_size = step_size or epsilon / 4
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.random_start = random_start

    def forward(self, model, x_natural, y, optimizer=None):
        model.eval()
        x_adv = x_natural.detach()

        if self.random_start:
            if self.distance == "L_inf":
                x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            elif self.distance == "L_2":
                delta = torch.randn_like(x_adv)
                delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                delta = delta / torch.clamp(delta_norm, min=1e-12)
                radius = torch.rand(x_adv.size(0), 1, 1, 1, device=x_adv.device)
                x_adv = x_adv + delta * radius * self.epsilon
            else:
                raise NotImplementedError(f"Unsupported distance: {self.distance}")

        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            if self.distance == "L_inf":
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            elif self.distance == "L_2":
                grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                x_adv = x_adv.detach() + self.step_size * grad.detach() / torch.clamp(grad_norm, min=1e-12)
                delta = x_adv - x_natural
                delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                factor = torch.min(torch.ones_like(delta_norm), self.epsilon / torch.clamp(delta_norm, min=1e-12))
                x_adv = x_natural + delta * factor

            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()
        optimizer.zero_grad(set_to_none=True) if optimizer is not None else None
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        return outputs, loss
