# basic_model.py (CPU-only safe)
import os
from collections import OrderedDict

import torch
from torchvision.transforms.functional import rgb_to_grayscale  # used in set_input

from . import networks
from .base_model import BaseModel


class CDEvaluator(BaseModel):
    def name(self):
        return "CDEvaluator"

    def __init__(self, args):
        # BaseModel will set device to CPU in the patched version
        BaseModel.__init__(self, args)
        self.loss_names = ["CE"]
        self.metric = ["acc"]
        self.visual_names = [
            "A_un",
            "A_tr",
            "A_un_gray",
            "A_tr_gray",
            "A_un_mask",
            "A_tr_mask",
            "real_A",
            "real_B",
            "fake_B",
            "rec_A",
            "real_A_mask",
        ]
        self.model_names = ["G"]

        # Force device to CPU (BaseModel already does this, but keep explicit here)
        self.device = torch.device("cpu")
        self.gpu_ids = []  # ensure no GPU usage

        # Create the network (networks.init_net will put it on CPU)
        self.net_G = networks.define_G(args=args, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.optimizer_G = torch.optim.Adam(
                self.net_G.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        # Move tensors to CPU device (safe)
        A_un = input["A_un"].to(self.device)
        A_tr = input["A_tr"].to(self.device)
        self.input = torch.cat([A_un, A_tr], 1)
        self.A_un = A_un
        self.A_tr = A_tr
        self.A_un_gray = rgb_to_grayscale(self.A_un)
        self.A_tr_gray = rgb_to_grayscale(self.A_tr)
        self.A_un_mask = self.A_un[:, 0:1, :, :]
        self.A_tr_mask = self.A_tr[:, 0:1, :, :]

        self.label = input["L"].to(self.device)
        self.image_paths = input["name"]

    def forward(self):
        # net_G expects CPU tensors now
        self.pred = self.net_G(self.A_un, self.A_tr)

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.loss_CE = self.criterionCE(self.pred, self.label.long())
        self.loss_CE.backward()
        self.optimizer_G.step()

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        visual_ret["A_un"] = self.A_un
        visual_ret["A_tr"] = self.A_tr
        visual_ret["real_A"] = self.input
        visual_ret["real_B"] = self.label
        visual_ret["fake_B"] = self.pred
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        errors_ret["CE"] = self.loss_CE.item()
        return errors_ret

    def get_current_metrics(self):
        metrics = OrderedDict()
        for name in self.metric:
            if isinstance(name, str):
                metrics[name] = getattr(self, "metric_" + name)
        return metrics

    def update_learning_rate(self):
        lrd = self.args.lr / self.args.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr
        print("update learning rate: %f -> %f" % (self.old_lr, lr))
        self.old_lr = lr

    def eval(self):
        # Put net in eval mode (no device change)
        self.net_G.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_network(self, network, network_label, epoch_label, gpu_ids=None):
        # Save CPU-only state dict
        if gpu_ids is None:
            gpu_ids = []
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        # Do NOT move back to CUDA on a CPU-only system

    def load_network(self, network, network_label, epoch_label):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Model checkpoint not found: {save_path}")

        # Always load onto CPU
        state_dict = torch.load(save_path, map_location="cpu")
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        network.load_state_dict(state_dict)
        # ensure network is on CPU
        network.to(torch.device("cpu"))
