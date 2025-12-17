import lightning as L
import segmentation_models_pytorch as smp

import torchmetrics
import torch
from torchmetrics.segmentation import MeanIoU, DiceScore
import torch.nn.functional as F
from typing import List


class ComboLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True, smooth=1)
        self.CE = torch.nn.CrossEntropyLoss()

    def forward(self, X, Y):
        dice = self.dice(X, Y)
        focal = self.CE(X, Y.squeeze())

        return dice +  focal

def fpn_embed_concat(feats:List[torch.Tensor]) -> torch.Tensor:
    """
    feats: list of 6 tensors, each [B, C_l, H_l, W_l]
    returns: [B, D] where D = sum(C_l), unnormalized, its done in the get_representation function
    """
    per_level = []
    for x in feats:
        # Global average pooling to [B, C_l]
        v_l = F.adaptive_avg_pool2d(x, 1)
        per_level.append(v_l)
    v = torch.cat(per_level, dim=1)
    return v.squeeze()


class SsegmentationMulticlass(L.LightningModule):
    def __init__(self, config, categories:List[str],num_channels:int=1):
        super().__init__()

        num_classes = len(categories)
        self.modelname:str = "fpn"
        self.model = smp.FPN(
            encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=num_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
        )
        self.loss_function = ComboLoss()
        self.categories:List[str] = categories

        metrics = torchmetrics.MetricCollection(
            torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
            MeanIoU(num_classes=num_classes, per_class=False, input_format="index"),
            torchmetrics.Recall(task="multiclass", num_classes=num_classes),
            torchmetrics.Precision(task="multiclass", num_classes=num_classes),
            DiceScore(num_classes=num_classes,input_format="index")
        )

        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        self.percategory_metric = torchmetrics.MetricCollection(
            MeanIoU(num_classes=num_classes, per_class=True, input_format="index"),
            torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average=None
            ),
            torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average=None
            ),
            torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average=None
            ),
            DiceScore(num_classes=num_classes, average=None, input_format="index")

        )
        self.config = config

    @staticmethod
    def prepare_batch(batch):
        return batch["image"], batch["mask"], batch["name"]

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def forward_proba(self, x):
        return self.model(x).softmax(dim=1)

    @torch.no_grad()
    def forward_predict(self, x):
        return self.model(x).argmax(1)

    @torch.no_grad()
    def forward_embeddings(self, x):
        return fpn_embed_concat(self.model.encoder(x))

    def activate_hook(self, features: List):
        """Activate hook, the goal is to be able to create embbeding matrix but also getting prediction, indeed now we can use either 
            forward_emb or forward but not have both at the same time and this is not practical, for example for BADGE strategy

        Args:
            features (List): _description_
        """        
        def hook_func(module, input, output):
            features.append(fpn_embed_concat(output).detach())

        handle = self.model.encoder.register_forward_hook(hook_func)

    def training_step(self, batch, batch_idx):
        img_b, target_b,_ = self.prepare_batch(batch)
        output = self.model(img_b)
        preds = output.argmax(1)
        loss = self.loss_function(output, target_b)

        # self.train_metrics(output, target_b)

        self.train_metrics(preds, target_b)

        self.log(
            "Train/Loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=output.shape[0],
        )

        optimizer = self.optimizers()
        self.log("Global/LR", optimizer.param_groups[0]["lr"])

        return {"loss": loss, "metric": self.train_metrics}

    def on_train_epoch_end(self):
        for metric_name, metric_object in self.train_metrics.compute().items():
            self.log(
                metric_name,
                metric_object,
                sync_dist=True,
                batch_size=self.config.batch_size,
            )

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        img_b, target_b,_ = self.prepare_batch(batch)

        output: torch.Tensor = self.model(img_b)
        preds = output.argmax(1)

        loss = self.loss_function(output, target_b)
        # Computed on output
        # self.val_metrics(output, target_b)
        # self.percategory_metric(output, target_b)

        # computed on preds
        self.val_metrics(preds, target_b)
        self.percategory_metric(preds, target_b)

        self.log(
            "Validation/Loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=output.shape[0],
        )

        return {"loss": loss}

    def on_validation_epoch_end(self):
        # logging metrics on a global basis
        for metric_name, metric_object in self.val_metrics.compute().items():
            self.log(
                metric_name,
                metric_object,
                sync_dist=True,
                batch_size=self.config.batch_size,
            )

        # Logging metrics on a category basis
        metric_dict = {}
        for metric_name, metric_object in self.percategory_metric.items():
            # we did not average the metrics over classes.
            # That is why we need to specifically log it
            for k, metric_value in enumerate(metric_object.compute()):
                metric_dict[f"Validation/{metric_name} {self.categories[k]}"] = (
                    metric_value
                )
                self.log(
                    f"Validation/{metric_name} {self.categories[k]}",
                    metric_value,
                    sync_dist=True,
                    batch_size=self.config.batch_size,
                )

        self.validation_metric_dict = metric_dict

        self.val_metrics.reset()
        self.percategory_metric.reset()
        return

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        img_b, target_b,_ = self.prepare_batch(batch)

        output = self.model(img_b)
        preds = output.argmax(1)

        loss = self.loss_function(output, target_b)

        # computed on preds
        self.test_metrics(preds, target_b)
        self.percategory_metric(preds.unsqueeze(1), target_b.unsqueeze(1))

        self.log(
            "Test/Loss", loss, on_epoch=True, prog_bar=True, batch_size=output.shape[0]
        )

        return {"loss": loss}

    def on_test_epoch_end(self):
        # logging metrics on a global basis
        for metric_name, metric_object in self.test_metrics.compute().items():
            self.log(
                metric_name,
                metric_object,
                sync_dist=True,
                batch_size=self.config.batch_size,
            )

        # Logging metrics on a category basis
        metric_dict = {}
        for metric_name, metric_object in self.percategory_metric.items():
            # we did not average the metrics over classes.
            # That is why we need to specifically log it
            for k, metric_value in enumerate(metric_object.compute()):
                metric_dict[f"Test/{metric_name} {self.categories[k]}"] = metric_value
                self.log(
                    f"Test/{metric_name} {self.categories[k]}",
                    metric_value,
                    sync_dist=True,
                    batch_size=self.config.batch_size,
                )


        self.test_metrics.reset()
        self.percategory_metric.reset()

        return metric_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=1e-4
        )

        return [optimizer]


if __name__ == "__main__":
    pass