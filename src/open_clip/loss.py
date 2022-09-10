import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


# def gather_features(
#         features,
#         local_loss=False,
#         gather_with_grad=False,
#         rank=0,
#         world_size=1,
#         use_horovod=False
# ):
#     assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
#     if features is None:
#         return None
#     if use_horovod:
#         assert hvd is not None, 'Please install horovod'
#         if gather_with_grad:
#             all_features = hvd.allgather(features)
#         else:
#             with torch.no_grad():
#                 all_features = hvd.allgather(features)
#             if not local_loss:
#                 # ensure grads for local rank when all_* features don't have a gradient
#                 gathered_features = list(all_features.chunk(world_size, dim=0))
#                 gathered_features[rank] = features
#                 all_features = torch.cat(gathered_features, dim=0)
#     else:
#         # We gather tensors from all gpus
#         if gather_with_grad:
#             all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
#         else:
#             gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
#             dist.all_gather(gathered_features, features)
#             if not local_loss:
#                 # ensure grads for local rank when all_* features don't have a gradient
#                 gathered_features[rank] = features
#             all_features = torch.cat(gathered_features, dim=0)
#
#     return all_features

class ClipLoss(nn.Module):

    def __init__(
            self,
            batch_size,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    # strategy1: using batch indices to select augmetation features
    # def forward(self, image_features, text_features, logit_scale):
    #     device = image_features.device
    #     if self.world_size > 1:
    #         all_image_features, all_text_features = gather_features(
    #             image_features, text_features,
    #             self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
    #
    #         if self.local_loss:
    #             logits_per_image = logit_scale * image_features[:self.batch_size] @ all_text_features.T
    #             logits_per_text = logit_scale * text_features[:self.batch_size] @ all_image_features.T
    #         else:
    #             _all_image_features = torch.cat([all_image_features[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, int(len(all_image_features) / self.batch_size), 2)], dim=0) if len(all_image_features) > self.batch_size * self.world_size else all_image_features
    #             _all_text_features = torch.cat([all_text_features[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, int(len(all_text_features) / self.batch_size), 2)], dim=0) if len(all_text_features) > self.batch_size * self.world_size else all_text_features
    #             logits_per_image = logit_scale * _all_image_features @ all_text_features.T
    #             logits_per_text = logit_scale * _all_text_features @ all_image_features.T
    #             # print(all_image_features.size(), all_text_features.size(), _all_image_features.size(), _all_text_features.size())
    #             # print(logits_per_image.size(), logits_per_text.size())
    #     else:
    #         logits_per_image = logit_scale * image_features[:self.batch_size] @ text_features.T
    #         logits_per_text = logit_scale * text_features[:self.batch_size] @ image_features.T
    #     print('image feature', image_features.size())
    #     print('image feature', image_features)
    #     print('text feature', text_features.size())
    #     print('text feature', text_features)
    #     print(logits_per_image.size(), logits_per_text.size())
    #     print('logits_per_image', logits_per_image)
    #     print('logits_per_image', logits_per_text)
    #
    #     # calculated ground-truth and cache if enabled
    #     num_logits = logits_per_image.shape[0]
    #     if self.prev_num_logits != num_logits or device not in self.labels:
    #         labels = torch.arange(num_logits, device=device, dtype=torch.long)
    #         if self.world_size > 1 and self.local_loss:
    #             labels = labels + num_logits * self.rank
    #         if self.cache_labels:
    #             self.labels[device] = labels
    #             self.prev_num_logits = num_logits
    #     else:
    #         labels = self.labels[device]
    #
    #     total_loss = (
    #         F.cross_entropy(logits_per_image, labels) +
    #         F.cross_entropy(logits_per_text, labels)
    #         ) / 2
    #     return total_loss


    # strategy2: split features and aug_features; set aug_features to None by default.
    # def forward(self, image_features, text_features, logit_scale, image_aug_features=None, text_aug_features=None):
    #     device = image_features.device
    #     if self.world_size >= 1:
    #         pdb.set_trace()
    #
    #         all_image_features = gather_features(image_features, self.local_loss, self.gather_with_grad, self.rank,
    #                                              self.world_size, self.use_horovod)
    #         all_text_features = gather_features(text_features, self.local_loss, self.gather_with_grad, self.rank,
    #                                              self.world_size, self.use_horovod)
    #         all_image_aug_features = gather_features(image_aug_features, self.local_loss, self.gather_with_grad, self.rank,
    #                                              self.world_size, self.use_horovod)
    #         all_text_aug_features = gather_features(text_aug_features, self.local_loss, self.gather_with_grad, self.rank,
    #                                              self.world_size, self.use_horovod)
    #
    #         _all_image_features = all_image_features if image_aug_features is None else torch.cat([all_image_features, all_image_aug_features], dim=0)
    #         _all_text_features = all_text_features if text_aug_features is None else torch.cat([all_text_features, all_text_aug_features], dim=0)
    #
    #         if self.local_loss:
    #             logits_per_image = logit_scale * image_features @ _all_text_features.T
    #             logits_per_text = logit_scale * text_features @ _all_image_features.T
    #         else:
    #             logits_per_image = logit_scale * all_image_features @ _all_text_features.T
    #             logits_per_text = logit_scale * all_text_features @ _all_image_features.T
    #     else:
    #         _image_features = image_features if image_aug_features is None else torch.cat([image_features, image_aug_features], dim=0)
    #         _text_features = text_features if text_aug_features is None else torch.cat([text_features, text_aug_features], dim=0)
    #         logits_per_image = logit_scale * image_features @ _text_features.T
    #         logits_per_text = logit_scale * text_features @ _image_features.T
    #
    #     # calculated ground-truth and cache if enabled
    #     num_logits = logits_per_image.shape[0]
    #     if self.prev_num_logits != num_logits or device not in self.labels:
    #         labels = torch.arange(num_logits, device=device, dtype=torch.long)
    #         if self.world_size > 1 and self.local_loss:
    #             labels = labels + num_logits * self.rank
    #         if self.cache_labels:
    #             self.labels[device] = labels
    #             self.prev_num_logits = num_logits
    #     else:
    #         labels = self.labels[device]
    #
    #     total_loss = (
    #         F.cross_entropy(logits_per_image, labels) +
    #         F.cross_entropy(logits_per_text, labels)
    #         ) / 2
    #     return total_loss


    # strategy3: split features and aug_features; set aug_features to blank tensor by default.
    def forward(self, image_features, text_features, logit_scale, image_aug_features=None, text_aug_features=None):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            all_image_aug_features, all_text_aug_features = gather_features(
                image_aug_features, text_aug_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ torch.cat([all_text_features, all_text_aug_features], dim=0).T
                logits_per_text = logit_scale * text_features @ torch.cat([all_image_features, all_image_aug_features], dim=0).T
            else:
                logits_per_image = logit_scale * all_image_features @ torch.cat([all_text_features, all_text_aug_features], dim=0).T
                logits_per_text = logit_scale * all_text_features @ torch.cat([all_image_features, all_image_aug_features], dim=0).T
        else:
            logits_per_image = logit_scale * image_features @ torch.cat([text_features, text_aug_features], dim=0).T
            logits_per_text = logit_scale * text_features @ torch.cat([image_features, image_aug_features], dim=0).T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        print(labels.device, labels.size())
        print(logits_per_image.device, logits_per_image.size())
        print(logits_per_image)
        print(logits_per_text.device, logits_per_text.size())
        print(logits_per_text)
        print(labels[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        print(total_loss)
        return total_loss




