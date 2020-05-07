import torch
import pdb


class AbstractLearning:
    """ Abstract class for learning """

    def __init__(self, model, calc_loss, optimizer, config, tensorboard=None):
        
        self.model = model
        self.config = config
        self.calc_loss = calc_loss
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.iter = 0
        self.grad_log_enable = False
        self.grad_log_iter = 200

    def do_update(self, replay_memory, logger):

        losses = self.calc_loss(replay_memory)

        loss = None
        if losses is not None:
            for loss_ in losses.values():
                if loss_ is not None:
                    if loss is None:
                        loss = loss_
                    else:
                        loss = loss + loss_
        else:
            return None


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

        if self.grad_log_enable:
            if self.iter % self.grad_log_iter == 0:
                self.write_grad_summaries()
        self.iter += 1

        loss = float(loss.item())

        return losses

    def write_grad_summaries(self):

        if self.tensorboard is None:
            return

        named_params = self.model.get_named_parameters()
        for name, parameter in named_params:
            weights = parameter.data.cpu()
            mean_weight = torch.mean(torch.abs(weights))
            weights = weights.numpy()
            self.tensorboard.log_histogram("hist_" + name + "_data", weights, bins=100)
            self.tensorboard.log_scalar("mean_" + name + "_data", mean_weight)
            if parameter.grad is not None:
                grad = parameter.grad.data.cpu()
                mean_grad = torch.mean(torch.abs(grad))
                grad = grad.numpy()
                self.tensorboard.log_histogram("hist_" + name + "_grad", grad, bins=100)
                self.tensorboard.log_scalar("mean_" + name + "_grad", mean_grad)
