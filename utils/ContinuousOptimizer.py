import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.optim import lr_scheduler


class ContinuousOptimizer:
    def __init__(self, client_grads, batch_size, server, client, device, tokenizer, lr, num_of_iter, labels,
                 attention_mask, decoder_attention_mask, spectrogram_lengths, transcript_lengths, alpha):
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
        self.attention_mask = attention_mask
        self.decoder_attention_mask = decoder_attention_mask
        self.spectrogram_lengths = spectrogram_lengths
        self.transcript_lengths = transcript_lengths

    def set_up(self):
        # Create permutations

        self.dummy_data = torch.randn(self.batch_size, 80, 3000).to(self.device).requires_grad_(True)
        self.labels = self.labels.to(self.device)

    def compute_gradient_distance(self, client_grads, server_grads):
        grads_diff = 0
        n_g = 0
        # for gx, gy in zip(server_grads, client_grads):
        #     grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)

        # for gx, gy in zip(server_grads, client_grads):
        #     grads_diff += torch.norm(gx - gy, p=2)

        # Calculate cosine distance
        for gx, gy in zip(server_grads, client_grads):
            grads_diff += 1.0 - (gx * gy).sum() / (gx.view(-1).norm(p=2) * gy.view(-1).norm(p=2))
            n_g += 1
        grads_diff /= n_g
        return grads_diff

    def optimize(self):
        self.set_up()
        optimizer = torch.optim.Adam([self.dummy_data], lr=self.lr)

        def lr_lambda(current_step: int):
            return max(0.0, float(self.num_of_iter - current_step) / float(max(1, self.num_of_iter)))

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for i in range(self.num_of_iter):
            dummy_outputs = self.server(input_features=self.dummy_data, labels=self.labels,
                                        attention_mask=self.attention_mask,
                                        decoder_attention_mask=self.decoder_attention_mask)
            dummy_loss = dummy_outputs.loss
            server_parameters = []
            for param in self.server.parameters():
                if param.requires_grad:
                    server_parameters.append(param)
            server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
            grads_diff = self.compute_gradient_distance(self.client_grads, server_grads)
            grads_diff.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            # Clamp the dummy data between -1 and 1
            self.dummy_data.data = torch.clamp(self.dummy_data.data, -1, 1)

            with torch.no_grad():
                for j in range(self.batch_size):
                    spectrogram_length = self.spectrogram_lengths[j]
                    self.dummy_data[j, :, spectrogram_length:] = 0

            if i % 100 == 0:
                print(i)
                print("Gradients difference: ", grads_diff.item())
            optimizer.zero_grad()
            self.server.zero_grad()
        return self.dummy_data.clone().detach()
