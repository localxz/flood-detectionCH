import os

import numpy as np
import torch
from misc.metric_tool import ConfuseMatrixMeter

from models.networks import define_G


class CDEvaluator:
    # --- START OF CORRECTION ---
    # The __init__ method is updated to accept the 'reporter' object.
    def __init__(self, args, dataloader, reporter=None):
        self.dataloader = dataloader
        self.n_class = args.n_class
        self.reporter = reporter

        self.device = torch.device("cpu")
        self._report(f"CDEvaluator is running on device: {self.device}")

        self.net_G = define_G(args=args, gpu_ids=[])
        self.net_G.to(self.device)

        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)
        self._report(str(args.__dict__))  # Replaces the old file logger

        # Training log state variables
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.steps_per_epoch = len(dataloader)
        self.G_pred = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _report(self, message):
        """A helper method to log messages to the reporter if it exists, otherwise print."""
        if self.reporter:
            # Add a prefix to easily identify logs from this class
            self.reporter.report(f"CD_EVAL: {message.strip()}")
        else:
            print(message.strip())

    # --- END OF CORRECTION ---

    def _load_checkpoint(self, checkpoint_name="best_ckpt.pt"):
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            # All print/logger calls are replaced with self._report
            self._report("Loading last checkpoint...")

            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Load state dict with strict=False to handle architecture changes
            self.net_G.load_state_dict(checkpoint["model_G_state_dict"], strict=False)
            self.net_G.to(self.device)

            self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
            self.best_epoch_id = checkpoint.get("best_epoch_id", 0)

            self._report(
                "Eval Historical_best_acc = %.4f (at epoch %d)"
                % (self.best_val_acc, self.best_epoch_id)
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * (255 // (self.n_class - 1))
        return pred_vis

    def _update_metric(self):
        """Updates the confusion matrix metric."""
        target = self.batch["L"].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(), gt=target.cpu().numpy()
        )
        return current_score

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloader)
        if self.batch_id % 100 == 1:
            message = "Is_training: %s. [%d/%d], running_mf1: %.5f" % (
                self.is_training,
                self.batch_id,
                m,
                running_acc,
            )
            self._report(message)

    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, "scores_dict.npy"), scores_dict)
        self.epoch_acc = scores_dict["mf1"]

        # This was just for creating an empty file, can be removed or kept
        with open(
            os.path.join(self.checkpoint_dir, f"{self.epoch_acc:.5f}.txt"), mode="a"
        ) as file:
            pass

        message = ""
        for k, v in scores_dict.items():
            message += f"{k}: {v:.5f} "
        self._report(message)

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch["A"].to(self.device)
        img_in2 = batch["B"].to(self.device)

        self.G_pred_64, self.G_pred_128, self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self):
        self._report("Begin evaluation...")
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
