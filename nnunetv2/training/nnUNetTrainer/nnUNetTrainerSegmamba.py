from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn

from monai.networks.nets import SwinUNETR
from nnunetv2.Network.SegMamba import SegMamba
# from nnunet_mednext import create_mednext_v1

class nnUNetTrainerSegmamba(nnUNetTrainerNoDeepSupervision):
    """
    Swin-UNETR default configuration
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # self.grad_scaler = None

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)
        model = SegMamba(in_chans=num_input_channels,out_chans=label_manager.num_segmentation_heads).cuda()

        return model
    
    # def train_step(self, batch: dict) -> dict:
    #     data = batch['data']
    #     target = batch['target']

    #     data = data.to(self.device, non_blocking=True)
    #     if isinstance(target, list):
    #         target = [i.to(self.device, non_blocking=True) for i in target]
    #     else:
    #         target = target.to(self.device, non_blocking=True)

    #     self.optimizer.zero_grad(set_to_none=True)
        
    #     output = self.network(data)
    #     l = self.loss(output, target)
    #     l.backward()
    #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
    #     self.optimizer.step()
        
    #     return {'loss': l.detach().cpu().numpy()}
    

    # def validation_step(self, batch: dict) -> dict:
    #     data = batch['data']
    #     target = batch['target']

    #     data = data.to(self.device, non_blocking=True)
    #     if isinstance(target, list):
    #         target = [i.to(self.device, non_blocking=True) for i in target]
    #     else:
    #         target = target.to(self.device, non_blocking=True)

    #     self.optimizer.zero_grad(set_to_none=True)

    #     # Autocast is a little bitch.
    #     # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    #     # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    #     # So autocast will only be active if we have a cuda device.
    #     output = self.network(data)
    #     del data
    #     l = self.loss(output, target)

    #     # the following is needed for online evaluation. Fake dice (green line)
    #     axes = [0] + list(range(2, output.ndim))

    #     if self.label_manager.has_regions:
    #         predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
    #     else:
    #         # no need for softmax
    #         output_seg = output.argmax(1)[:, None]
    #         predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
    #         predicted_segmentation_onehot.scatter_(1, output_seg, 1)
    #         del output_seg

    #     if self.label_manager.has_ignore_label:
    #         if not self.label_manager.has_regions:
    #             mask = (target != self.label_manager.ignore_label).float()
    #             # CAREFUL that you don't rely on target after this line!
    #             target[target == self.label_manager.ignore_label] = 0
    #         else:
    #             mask = 1 - target[:, -1:]
    #             # CAREFUL that you don't rely on target after this line!
    #             target = target[:, :-1]
    #     else:
    #         mask = None

    #     tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

    #     tp_hard = tp.detach().cpu().numpy()
    #     fp_hard = fp.detach().cpu().numpy()
    #     fn_hard = fn.detach().cpu().numpy()
    #     if not self.label_manager.has_regions:
    #         # if we train with regions all segmentation heads predict some kind of foreground. In conventional
    #         # (softmax training) there needs tobe one output for the background. We are not interested in the
    #         # background Dice
    #         # [1:] in order to remove background
    #         tp_hard = tp_hard[1:]
    #         fp_hard = fp_hard[1:]
    #         fn_hard = fn_hard[1:]

    #     return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    # # def configure_optimizers(self):

    # #     optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
    # #     scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

    # #     self.print_to_log_file(f"Using optimizer {optimizer}")
    # #     self.print_to_log_file(f"Using scheduler {scheduler}")

    # #     return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass
    
    def save_network_architecture(self):
    # 注释掉原始代码以禁用保存
        pass