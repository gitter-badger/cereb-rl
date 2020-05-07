import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cuda import cuda_var


class PPOModel(nn.Module):

    def __init__(self, config, constants):
        super(PPOModel, self).__init__()

        self.prob_layer = nn.Sequential(
            nn.Linear(config["obs_dim"], config["num_actions"])
        )

        self.v_layer = nn.Sequential(
            nn.Linear(config["obs_dim"], 1)
        )

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x, action=None):
        # Assumes a batch size of 1
        
        logits = self.prob_layer(x)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1).mean()
        v = self.v_layer(x)

        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
              action = dist.sample()
        log_prob = log_probs.gather(1, action.view(-1, 1))

        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}
