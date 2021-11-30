import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #self.save_hyperparameters(kwargs)
        # load an instance segmentation model pre-trained on COCO
        if "inference_params" in kwargs:
            inference_params = kwargs.get("inference_params")
            self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                               **inference_params)
        else:
            self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.hparams.num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.detector.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self.hparams.hidden_size
        # and replace the mask predictor with a new one
        self.detector.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                   hidden_layer,
                                                                   self.hparams.num_classes)
        # These are initialised at the beginning of the training process
        self.valid_evaluator = None
        self.test_evaluator = None

    def configure_optimizers(self):
        # construct an optimizer
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.hparams.lr,
                                    momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=self.hparams.step_size,
                                                       gamma=self.hparams.gamma)

        return [optimizer], [lr_scheduler]

    def forward(self, image, target):
        loss_dict = self.detector(image, target)

        return loss_dict

    def on_validation_start(self) -> None:
        if self.valid_evaluator is None:
            coco = get_coco_api_from_dataset(self.val_dataloader().dataset, "validation")
            iou_types = _get_iou_types(self.detector)
            self.valid_evaluator = CocoEvaluator(coco, iou_types)
        self.valid_evaluator.reset()

    def on_test_start(self) -> None:
        if self.test_evaluator is None:
            iou_types = _get_iou_types(self.detector)
            coco = get_coco_api_from_dataset(self.test_dataloader().dataset, "test")
            self.test_evaluator = CocoEvaluator(coco, iou_types)
        self.test_evaluator.reset()

    def on_validation_end(self) -> None:
        self.valid_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        self.valid_evaluator.accumulate()
        self.valid_evaluator.summarize()

    def on_test_end(self) -> None:
        self.test_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        self.test_evaluator.accumulate()
        self.test_evaluator.summarize()

    def training_step(self, batch, batch_idx):
        image, target = batch
        loss_dict = self.forward(image, target)

        losses = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict, on_epoch=True)

        return losses

    def inference_step(self, batch, split_key):
        image, target = batch
        outputs = self.detector(image)

        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(target, outputs)}
        if split_key == "val":
            self.valid_evaluator.update(res)
        else:
            self.test_evaluator.update(res)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, "test")

