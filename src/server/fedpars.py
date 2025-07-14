import os
import time
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.base import get_acc_avg, printlog_stats, get_weights, set_clients , aggregate_parameters
from utils import optimizers
from client.client_fedpars import Client

class FedPARS():
    def __init__(self, args, model):
        self.start_time = time.time()
        self.args = args
        # Load datasets and splits
        self.acc_types = ['train', 'valid']
        self.p_clients = None
        self.m = max(int(self.args.frac_clients * self.args.num_clients), 1)
        self.last_round = -1
        self.iter = 0
        self.global_protos = []

        self.optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
        self.model = model

    def train(self):
        clients = set_clients(self.args,Client)
        torch.use_deterministic_algorithms(False)
        # Log experiment summary, client distributions, example images

        logger = SummaryWriter(f'./runs/{self.args.name}')
        input_size = (1,) + tuple([3, 512, 512])
        fake_input = torch.zeros(input_size).to(self.args.device)
        logger.add_graph(self.model, fake_input)

        acc_avg = get_acc_avg(self.acc_types, clients, self.model, self.args.device)
        acc_avg_best = acc_avg[self.acc_types[1]]

        # Print and log initial stats
        if not self.args.quiet:
            print('Training:')
        loss_avg, lr = torch.nan, torch.nan

        init_end_time = time.time()

        # number of communication rounds
        for round in range(self.last_round + 1, self.args.rounds):
            if not self.args.quiet:
                print(f'    Round: {round + 1}' + (f'/{self.args.rounds}' if self.args.iters is None else ''))

            # Sample clients每次选10个
            client_ids = np.random.choice(range(self.args.num_clients), self.m, replace=False, p=self.p_clients)

            # Train client models
            updates, num_examples, max_iters, loss_tot = [], [], 0, 0.
            client_examples_list = []
            local_protos = []
            for i, client_id in enumerate(client_ids):
                if not self.args.quiet:
                    print(f'        Client: {client_id} ({i + 1}/{self.m})')

                client_model = deepcopy(self.model)
                self.optimizer.__setstate__({'state': defaultdict(dict)})
                self.optimizer.param_groups[0]['params'] = list(client_model.parameters())

                client_update, client_num_examples, client_num_iters, client_loss, protos = clients[client_id].train(
                    model=client_model, optim=self.optimizer, device=self.args.device, global_protos=self.global_protos)
                client_examples_list.append(client_num_examples)

                local_protos.append(protos)
                if client_num_iters > max_iters: max_iters = client_num_iters

                if client_update is not None:
                    updates.append(deepcopy(client_update))
                    loss_tot += client_loss * client_num_examples
                    num_examples.append(client_num_examples)

            self.iter += max_iters
            lr = self.optimizer.param_groups[0]['lr']
            # weights_list = get_weights(local_protos)
            weights_list = [client_examples / sum(client_examples_list) for client_examples in client_examples_list]

            if len(updates) > 0:
                aggregate_parameters(self.args,self.model, updates, num_examples, weights_list)
                # Compute round average loss and accuracies
                if round % self.args.server_stats_every == 0:
                    loss_avg = loss_tot / sum(num_examples)
                    # IoU
                    acc_avg = get_acc_avg(self.acc_types, clients, self.model, self.args.device)

                    if acc_avg[self.acc_types[1]] > acc_avg_best:
                        acc_avg_best = acc_avg[self.acc_types[1]]

            if round % self.args.server_stats_every == 0:
                printlog_stats(self.args.quiet, logger, loss_avg, acc_avg, self.acc_types, lr, round + 1, self.iter, self.args.iters)

            # Stop training if the desired number of iterations has been reached
            if self.args.iters is not None and self.iter >= self.args.iters: break

        train_end_time = time.time()
        # Save model
        os.makedirs('./snapshot', exist_ok=True)
        torch.save(self.model.state_dict(), "./snapshot/" + self.args.name + ".pth")

        test_end_time = time.time()

        print(f'    Train time: {timedelta(seconds=int(train_end_time - init_end_time))}')
        print(f'    Total time: {timedelta(seconds=int(time.time() - self.start_time))}')

        if logger is not None:
            logger.close()