from torch.optim import Adam

from autoinit import AutoInitializer

## Taken from: https://github.com/gordicaleksa/pytorch-original-transformer/blob/main/utils/optimizers_and_distributions.py#L5
@AutoInitializer
class CustomLRAdamOptimizer:
    def __init__(self, optimizer, d_model, num_warmup_steps):
        self.current_step_number = 0

    def __repr__(self): 
      return f'Custom-LR Adam optimizer w/ optimizer={self.optimizer}, ' \
           + f'd_model={self.d_model}, ' \
           + f'and num_warmup_steps={self.num_warmup_steps}'

    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.d_model**(-.5) * min(
                     self.current_step_number**(-.5), 
                     self.current_step_number * self.num_warmup_steps**(-1.5))

        for group in self.optimizer.param_groups: 
          group['lr'] = current_learning_rate

        self.optimizer.step()

    def zero_grad(self): self.optimizer.zero_grad()

def get_optimizer(model, num_warmup_steps): return (optimizer := 
  CustomLRAdamOptimizer(Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), 
                        model.d_model, num_warmup_steps))




