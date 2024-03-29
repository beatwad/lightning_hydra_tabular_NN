from typing import Any, Dict, Tuple

import lightning as L
import torch
from torch import nn
from torchmetrics import AveragePrecision, Accuracy, MaxMetric, MeanMetric


# define the LightningModule
class TabularModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        data_config: dict,
    ) -> None:
        """Initialize a `TabularModule` for tabular data.

        Parameters
        ----------
        data_config
            Configuration of the data (input width, batch size, etc).
        model
            PyTorch model to train.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # params
        batch_size = data_config["batch_size"]

        # model
        self.model = model
        input_width = self.model.input_width

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging AP across batches
        self.train_metric = AveragePrecision(task="binary")
        self.val_metric = AveragePrecision(task="binary")
        self.test_metric = AveragePrecision(task="binary")

        # for averaging loss across batchess
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation AP
        self.val_metric_best = MaxMetric()

        # use this to show input dimensions of the models
        self.example_input_array = torch.Tensor(batch_size, input_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.model`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric.reset()
        self.val_metric_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, logits, targets = self.model_step(batch)
        # softmax should be here, if it's placed to __init__() - CUDA error occurs (don't know why)
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)[:, 1]

        # update and log metrics
        self.train_loss(loss)
        self.train_metric(probs, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ap", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, logits, targets = self.model_step(batch)
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)[:, 1]

        # update and log metrics
        self.val_loss(loss)
        self.val_metric(probs, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ap", self.val_metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        ap = self.val_metric.compute()  # get current val_metric
        self.val_metric_best(ap)  # update best so far val_metric
        # log `val_metric_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/ap_best", self.val_metric_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and targetgit 
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, logits, targets = self.model_step(batch)
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)[:, 1]

        # update and log metrics
        self.test_loss(loss)
        self.test_metric(probs, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ap", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
