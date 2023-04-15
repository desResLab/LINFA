import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import path

torch.set_default_tensor_type(torch.DoubleTensor)


class FNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x


class Surrogate:
    def __init__(self, model_name, model_func, input_size, output_size, limits=None, memory_len=20):
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = model_name
        self.mf = model_func
        self.pre_out = None
        self.m = None
        self.sd = None
        self.tm = None
        self.tsd = None
        self.limits = limits
        self.pre_grid = None
        self.surrogate = FNN(input_size, output_size)
        self.beta_0 = 0.5
        self.beta_1 = 0.1

        self.memory_grid = []
        self.memory_out = []
        self.memory_len = memory_len
        self.weights = torch.Tensor([np.exp(-self.beta_1 * i) for i in range(memory_len)])
        self.grid_record = None

    @property
    def limits(self):
        return self.__limits

    @limits.setter
    def limits(self, limits):
        limits = torch.Tensor(limits).tolist()
        if len(limits) != self.input_size:
            print("Error: Invalid input size for limit. Abort.")
            exit(-1)
        elif any(len(item) != 2 for item in limits):
            print("Error: Limits should be two bounds. Abort.")
            exit(-1)
        elif any(item[0] > item[1] for item in limits):
            print("Error: Upper bound should not be smaller than lower bound. Abort.")
            exit(-1)
        self.__limits = limits

    @property
    def pre_grid(self):
        return self.__pre_grid

    @pre_grid.setter
    def pre_grid(self, pre_grid):
        if pre_grid is None:
            if path.exists(self.model_name + '.npz'):
                container = np.load(self.model_name + '.npz')
                if 'pre_grid' in container:
                    self.pre_grid = container['pre_grid']
                    print("Success: Pre-Grid found.")
            else:
                print("Warning: " + self.model_name + ".npz does not found, please generate pre-grid.")
                print("Suggestion: Use Surrogate.gen_grid(input_limits=None, grid_num=5, store=True)")
        else:
            self.__pre_grid = torch.Tensor(pre_grid)
            self.m = torch.mean(self.pre_grid, 0)
            self.sd = torch.std(self.pre_grid, 0)
            self.pre_out = self.mf(self.pre_grid)
            self.tm = torch.mean(self.pre_out, 0)
            self.tsd = torch.std(self.pre_out, 0)
            self.grid_record = self.__pre_grid.clone()

    def gen_grid(self, input_limits=None, gridnum=4, store=True):
        meshpoints = []
        if input_limits is not None:
            self.limits = input_limits
            print("Warning: Input limits recorded in surrogate.")

        for lim in self.limits: meshpoints.append(torch.linspace(lim[0], lim[1], steps=gridnum))
        grid = torch.meshgrid(meshpoints)
        grid = torch.cat([item.reshape(gridnum ** len(self.limits), 1) for item in grid], 1)
        if store:
            self.pre_grid = grid
            self.grid_record = self.pre_grid.clone()
            self.surrogate_save()
        return grid

    def surrogate_save(self):
        torch.save(self.surrogate.state_dict(), self.model_name + '.sur')
        np.savez(self.model_name, limits=self.limits, pre_grid=self.pre_grid,
                 grid_record=self.grid_record)

    def surrogate_load(self):
        self.surrogate.load_state_dict(torch.load(self.model_name + '.sur'))
        container = np.load(self.model_name + '.npz')
        for key in container:
            try:
                setattr(self, key, torch.Tensor(container[key]))
                print("Success: [" + key + "] loaded.")
            except:
                print("Warning: [" + key + "] is not a surrogate variables.")

    def pre_train(self, max_iters, lr, lr_exp, record_interval, store=True, reg=False):
        grid = (self.pre_grid - self.m) / self.sd
        out = (self.pre_out - self.tm) / self.tsd
        optimizer = torch.optim.RMSprop(self.surrogate.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_exp)
        for i in range(max_iters):
            self.surrogate.train()
            scheduler.step()
            y = self.surrogate(grid)
            loss = torch.sum((y - out) ** 2) / y.size(0)
            if reg:
                reg_loss = 0
                for param in self.surrogate.parameters():
                    reg_loss += torch.abs(param).sum() * 0.0001
                loss += reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % record_interval == 0:
                if reg:
                    print('iter {}\t loss {}\t reg_loss {}'.format(i, loss, reg_loss))
                else:
                    print('iter {}   loss {}'.format(i, loss))
        if store: self.surrogate_save()

    def update(self, x, max_iters=10000, lr=0.01, lr_exp=0.999, record_interval=500, store=False, tol=1e-5, reg=False):
        self.grid_record = torch.cat((self.grid_record, x), dim=0)
        s = torch.std(x, dim=0)
        thresh = 0.1
        if torch.any(s < thresh):
            p = x[:, s < thresh]
            x[:, s < thresh] += torch.normal(0, 1, size=tuple(p.size())) * thresh
        print("Std: ", s)
        print("Std after: ", torch.std(x, dim=0))
        if len(self.memory_grid) >= self.memory_len:
            self.memory_grid.pop()
            self.memory_out.pop()
        self.memory_grid.insert(0, (x - self.m) / self.sd)
        self.memory_out.insert(0, (self.mf(x) - self.tm) / self.tsd)
        sizes = [list(self.pre_grid.size())[0]] + [list(item.size())[0] for item in self.memory_grid]
        # print(sizes)
        optimizer = torch.optim.RMSprop(self.surrogate.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_exp)
        for i in range(max_iters):
            self.surrogate.train()
            scheduler.step()
            y = self.surrogate(torch.cat(((self.pre_grid - self.m) / self.sd, *self.memory_grid), dim=0))
            out = torch.cat(((self.pre_out - self.tm) / self.tsd, *self.memory_out), dim=0)
            raw_loss = torch.stack([item.mean() for item in torch.split(torch.sum((y - out) ** 2, dim=1), sizes)])
            loss = raw_loss[0] * 2 * self.beta_0 * self.weights[:len(self.memory_grid)].sum() + torch.sum(
                raw_loss[1:] * self.weights[:len(self.memory_grid)]) * (1 - self.beta_0) * 2

            # loss = raw_loss[0] * self.weights[:len(self.memory_grid)].sum() + torch.sum(
            #     raw_loss[1:] * self.weights[:len(self.memory_grid)])
            if reg:
                for param in self.surrogate.parameters():
                    loss += torch.abs(param).sum() * 0.1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % record_interval == 0:
                print('Updating: {}\t loss {}'.format(i, loss), end='\r')
            if loss < tol ** 2: break
        print('                                                        ', end='\r')
        if store: self.surrogate_save()

    def forward(self, x):
        return self.surrogate((x - self.m) / self.sd) * self.tsd + self.tm


if __name__ == '__main__':
    S = Surrogate("Trivial", None, 3, 2, [[-7, 7], [-7, 7], [-7, 7]], 10, "Trivial_pregrid.txt")
    print(S.pre_grid)
