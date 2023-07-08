from .waseda import Cheng2020Attention
import torch.nn as nn
import torch

from compressai.layers import MaskedConv2d
from compressai.entropy_models.entropy_models import GaussianMixtureConditional
from compressai.registry import register_model


class MultistageSpatialContext(nn.Module):
    def __init__(self, patch_size, pattern, M=192, kernel_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcm = nn.ModuleList([])

        self.patch_size = patch_size
        self.step_count = patch_size ** 2
        self.M = M

        self.pattern = pattern

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.register_buffer('prev_latent_mask',
                             torch.zeros([self.step_count, 1, 1, patch_size, patch_size], requires_grad=False))
        self.register_buffer('curr_latent_mask',
                             torch.zeros([self.step_count, 1, 1, patch_size, patch_size], requires_grad=False))

        # Figure out context mask of each decoding step on a scratch board.
        board_size = self.padding * 2 + patch_size
        board = torch.zeros((board_size, board_size))
        for i in range(self.step_count):
            x, y = self.get_xy_offset(i)

            context_mask = self._get_context_mask(board, x, y)
            board = self._fill_board(board, x, y)
            self.gcm.append(MaskedConv2d(M, M * 2, mask_type=context_mask,
                                         kernel_size=kernel_size, stride=1, padding=self.padding, bias=False))

            self.prev_latent_mask[i+1:self.step_count, 0, 0, x, y] = 1
            self.curr_latent_mask[i, 0, 0, x, y] = 1

    def forward(self, y, step):
        mask = self.prev_latent_mask[step] \
            .repeat(1, 1, y.size(2) // self.patch_size, y.size(3) // self.patch_size) \
            .expand(y.size(0), y.size(1), -1, -1)
        masked_context = self.gcm[step](y * mask)

        context_mask = self.curr_latent_mask[step]. \
            repeat(1, 1, masked_context.size(2) // self.patch_size, masked_context.size(3) // self.patch_size) \
            .expand(masked_context.size(0), masked_context.size(1), -1, -1)
        return masked_context * context_mask

    def get_xy_offset(self, step):
        return (self.pattern == step).nonzero().squeeze().tolist()

    def _get_context_mask(self, board, x, y):
        context_mask = board[x:x+self.kernel_size, y:y+self.kernel_size]
        context_mask[self.kernel_size // 2, self.kernel_size // 2] = 0
        return context_mask

    def _fill_board(self, board, x, y):
        offsets = (-self.patch_size, 0, self.patch_size)
        for offset_x in offsets:
            for offset_y in offsets:
                curr_x = self.padding + x + offset_x
                curr_y = self.padding + y + offset_y
                if curr_x < 0 or curr_x >= board.size(0) or curr_y < 0 or curr_y >= board.size(1):
                    continue

                board[curr_x, curr_y] = 1

        return board


class Cheng2020AttentionGMMMultistage(Cheng2020Attention):
    def __init__(self, patch_size, context_pattern, M=192, K=3):
        super().__init__(M)
        self.patch_size = patch_size
        self.M = M
        self.K = K

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, 3 * self.M * self.K, 1),
        )
        self.gaussian_conditional = GaussianMixtureConditional(K=self.K)

        self.multistage_context = MultistageSpatialContext(patch_size, context_pattern, M)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        hyper_info = self.h_s(z_hat)

        entropy_params_size = (y.size(0), self.K * self.M, y.size(2), y.size(3))
        scales_all = torch.zeros(entropy_params_size, device=x.device)
        means_all = torch.zeros(entropy_params_size, device=x.device)
        weights_all = torch.zeros(entropy_params_size, device=x.device)

        for i in range(self.multistage_context.step_count):
            if i != 0:
                masked_context = self.multistage_context.forward(y_hat, i)
            else:
                masked_context = torch.zeros_like(hyper_info)

            gaussian_params = self.entropy_parameters(
                torch.cat((hyper_info, masked_context), dim=1)
            )

            scales, means, weights = gaussian_params.chunk(3, 1)
            mask = self.multistage_context.curr_latent_mask[i].\
                repeat(1, 1, scales.size(2) // self.patch_size, scales.size(3) // self.patch_size) \
                .expand(scales.size(0), scales.size(1), -1, -1)

            scales = scales * mask
            means = means * mask
            weights = weights * mask

            scales_all += scales
            means_all += means
            weights_all += weights

        weights_all = self._reshape_gmm_weight(weights_all)
        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means_all, weights_all)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_info = self.h_s(z_hat)
        y_hat = torch.zeros_like(y)
        strings = []

        for i in range(self.multistage_context.step_count):
            x_offset, y_offset = self.multistage_context.get_xy_offset(i)

            if i != 0:
                masked_context = self.multistage_context.forward(y_hat, i)
            else:
                masked_context = torch.zeros_like(hyper_info)

            gaussian_params = self.entropy_parameters(
                torch.cat((hyper_info, masked_context), dim=1)
            )
            scales, means, weights = gaussian_params.chunk(3, 1)
            scales = scales[:, :, x_offset::self.patch_size, y_offset::self.patch_size]
            means = means[:, :, x_offset::self.patch_size, y_offset::self.patch_size]
            weights = self._reshape_gmm_weight(
                weights[:, :, x_offset::self.patch_size, y_offset::self.patch_size])

            y_string, y_curr_split = self.gaussian_conditional.compress(
                y[:, :, x_offset::self.patch_size, y_offset::self.patch_size],
                scales, means, weights)

            y_hat[:, :, x_offset::self.patch_size, y_offset::self.patch_size] += y_curr_split
            strings.append(y_string)

        strings.append(z_strings)

        return {"strings": strings,
                "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        device = next(self.parameters()).device
        y_hat = torch.zeros([1, self.M, shape[0] * 4, shape[1] * 4]).to(device)

        z_hat = self.entropy_bottleneck.decompress(strings[-1], shape)
        hyper_info = self.h_s(z_hat)

        for i in range(self.multistage_context.step_count):
            x_offset, y_offset = self.multistage_context.get_xy_offset(i)

            if i != 0:
                masked_context = self.multistage_context.forward(y_hat, i)
            else:
                masked_context = torch.zeros_like(hyper_info)

            gaussian_params = self.entropy_parameters(
                torch.cat((hyper_info, masked_context), dim=1)
            )
            scales, means, weights = gaussian_params.chunk(3, 1)
            scales = scales[:, :, x_offset::self.patch_size, y_offset::self.patch_size]
            means = means[:, :, x_offset::self.patch_size, y_offset::self.patch_size]
            weights = self._reshape_gmm_weight(
                weights[:, :, x_offset::self.patch_size, y_offset::self.patch_size])

            y_curr_split = self.gaussian_conditional.decompress(*strings[i], scales, means, weights)
            y_hat[:, :, x_offset::self.patch_size, y_offset::self.patch_size] = y_curr_split

        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def _reshape_gmm_weight(self, weight):
        weight = torch.reshape(weight, (weight.size(0), self.K, weight.size(1) // self.K, weight.size(2), -1))
        weight = nn.functional.softmax(weight, dim=1)
        weight = torch.reshape(weight, (weight.size(0), weight.size(1) * weight.size(2), weight.size(3), -1))
        return weight


@register_model("cheng2020-attention-gmm-multistage-2x2")
class Cheng2020AttentionGMMMultistage2x2(Cheng2020AttentionGMMMultistage):
    def __init__(self, M=192, K=3):
        pattern = torch.tensor((
            [ 0, 1 ],
            [ 2, 3 ]))
        super().__init__(2, pattern, M, K, )


@register_model("cheng2020-attention-gmm-multistage-4x4")
class Cheng2020AttentionGMMMultistage4x4(Cheng2020AttentionGMMMultistage):
    def __init__(self, M=192, K=3):
        pattern = torch.tensor((
            [ 0,  2,  5,  4 ],
            [ 1,  7, 11,  8 ],
            [ 6, 12, 13, 10 ],
            [ 3, 14, 15,  9 ]))
        super().__init__(4, pattern, M, K)