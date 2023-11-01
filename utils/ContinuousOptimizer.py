import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot

class ContinuousOptimizer:
    def __init__(self, client_grads, batch_size, server, client, device, tokenizer, lr, num_of_iter, labels, alpha):
        self.one_hot_labels = None
        self.labels = labels
        self.alpha = alpha
        self.dummy_data = None
        self.client_grads = client_grads
        self.batch_size = batch_size
        self.server = server
        self.client = client
        self.device = device
        self.tokenizer = tokenizer
        self.lr = lr
        self.num_of_iter = num_of_iter
    
    def set_up(self):
        self.dummy_data = torch.randn(self.batch_size, 80, 3000).to(self.device).requires_grad_(True)
        self.labels = self.labels.to(self.device)
        self.one_hot_labels = torch.tensor(one_hot(self.labels, num_classes=len(self.tokenizer)),
                               dtype=torch.float32).to(self.device)

    def compute_gradient_distance(self, client_grads, server_grads):
        grads_diff = 0
        for gx, gy in zip(server_grads, client_grads):
            grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
        return grads_diff

    def optimize(self):
        self.set_up()
        optimizer = torch.optim.AdamW([self.dummy_data], lr=self.lr)
        for i in range(self.num_of_iter):
            dummy_outputs = self.server(input_features=self.dummy_data, labels=self.labels)
            dummy_preds = dummy_outputs.logits
            loss_function = torch.nn.CrossEntropyLoss()
            shifted_preds = dummy_preds[..., :-1, :].contiguous()
            shifted_labels = F.softmax(self.one_hot_labels, dim=-1)[..., 1:, :].contiguous()
            flatten_shifted_preds = shifted_preds.view(-1, shifted_preds.size(-1)).to(self.device)
            flatten_labels = shifted_labels.view(-1, shifted_labels.size(-1)).to(self.device)
            dummy_loss = loss_function(flatten_shifted_preds, flatten_labels)
            server_parameters = []
            for param in self.server.parameters():
                if param.requires_grad:
                    server_parameters.append(param)
            server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
            grads_diff = self.compute_gradient_distance(self.client_grads, server_grads)
            grads_diff.backward(retain_graph=True)
            optimizer.step()
            if i % 100 == 0:
                print(i)
                print("Gradients difference: ", grads_diff.item())
            optimizer.zero_grad()
            self.server.zero_grad()
        return self.dummy_data.clone().detach()