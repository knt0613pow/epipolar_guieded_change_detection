import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as tf
import numpy as np
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw
device = torch.device("cuda")
from typing import Dict, List, Optional, Tuple
from torchvision.models.detection.roi_heads import fastrcnn_loss
"""
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
"""

class FasterRCNN(nn.Module):
    def __init__(self, Weight_path, batch_norm=True, final_layers=False ):
        super(FasterRCNN, self).__init__()
        self.transform, self.backbone, self.rpn, self.roi_head = self.get_backbone(Weight_path)

    @staticmethod
    def get_backbone(Weight_path):
        model = models.detection.fasterrcnn_resnet50_fpn(num_classes  = 3,pretrained=False)
        if Weight_path != False:
            model.load_state_dict(torch.load(Weight_path))
        return model.transform,model.backbone , model.rpn, model.roi_heads

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections
        

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.roi_head.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        box_features = self.roi_head.box_roi_pool(features, proposals, images.image_sizes)

        box_features = self.roi_head.box_head(box_features)
        class_logits, box_regression = self.roi_head.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_box_reg": loss_box_reg
            }
            box, scores, labels = self.roi_head.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            ROI_feature = self.roi_head.box_roi_pool        (features, box, images.image_sizes)
            ROI_feature = self.roi_head.box_head(ROI_feature)
            num_images = len(box)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": box[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        else:
            box, scores, labels = self.roi_head.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            num_images = len(box)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": box[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
            ROI_feature = self.roi_head.box_roi_pool(features, box, images.image_sizes)
            ROI_feature = self.roi_head.box_head(ROI_feature)

        detection, detection_loss = result, losses

        res_loss = {}
        res_loss.update(detection_loss)
        res_loss.update(proposal_losses)


        return ROI_feature, res_loss, detection

