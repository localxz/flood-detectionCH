import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch


class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
        # Force CPU usage
        self.device = torch.device("cpu")
        self.gpu_ids = []  # No GPU IDs for DataParallel
        self.isTrain = args.isTrain

        self.save_dir = os.path.join(args.checkpoint_dir, args.name)
        if args.preprocess != "scale_width":
            self.visual_names = []
        self.model_names = []
        self.optimizers = []
        self.metric = 0

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, args):
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, args) for optimizer in self.optimizers
            ]
        if not self.isTrain or args.continue_train:
            self.load_networks(args.epoch)
        self.print_networks(args.verbose)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name).data
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, "loss_" + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)
                torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from %s" % load_path)
                state_dict = torch.load(load_path, map_location="cpu")
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = sum(p.numel() for p in net.parameters())
                if verbose:
                    print(net)
                print(
                    "[Network %s] Total number of parameters : %.3f M"
                    % (name, num_params / 1e6)
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
