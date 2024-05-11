#微调decoder


import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import utils
import libs_img
import libs_model

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#读取传入参数
def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, help="Path to the training data directory", required=True)
    aa("--val_dir", type=str, help="Path to the validation data directory", required=True)

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, help="Path to the configuration file for the LDM model")
    aa("--ldm_ckpt", type=str, help="Path to the checkpoint file for the LDM model")
    aa("--msg_decoder_path", type=str, help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg",
       help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser

#主函数
def main(params):
    # 设置随机种子
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

    # 打印出传入参数
    print("__log__:{}".format(json.dumps(vars(params))))

    # 创建输出文件夹
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    params.imgs_dir = imgs_dir
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # 加载LDM auto-encoder models
    print(f'调用LDM模型。 config:{params.ldm_config} weights:{params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = libs_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    ldm_ae.eval()
    ldm_ae.to(device)

    # 是否有已经白化处理过的参数
    print(f'调用hidden decoder。weights：{params.msg_decoder_path}...')
    if 'torchscript' in params.msg_decoder_path:
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        # 加载完毕

    else:
        # 否则进行白化处理
        msg_decoder = libs_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy,
                                                     num_blocks=params.decoder_depth,
                                                     channels=params.decoder_channels).to(device)
        ckpt = libs_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
        msg_decoder.eval()

        #加载参数
        print(f'加载参数...')
        with torch.no_grad():
            #
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            loader = utils.get_dataloader(params.train_dir, transform, batch_size=16, collate_fn=None)
            ys = []
            for i, x in enumerate(loader):
                x = x.to(device)
                y = msg_decoder(x)
                ys.append(y.to('cpu'))
            ys = torch.cat(ys, dim=0)
            nbit = ys.shape[1]

            # 训练一个线性层，然后将训练好的线性层放到原模型的最后，达到白化效果
            mean = ys.mean(dim=0, keepdim=True)  # NxD -> 1xD
            ys_centered = ys - mean  # NxD
            cov = ys_centered.T @ ys_centered
            e, v = torch.linalg.eigh(cov)
            L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
            weight = torch.mm(L, v.T)
            bias = -torch.mm(mean, weight.T).squeeze(0)
            linear = nn.Linear(nbit, nbit, bias=True)
            linear.weight.data = np.sqrt(nbit) * weight
            linear.bias.data = np.sqrt(nbit) * bias
            msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
            torchscript_m = torch.jit.script(msg_decoder)
            params.msg_decoder_path = params.msg_decoder_path.replace(".pth", "_whit.pth")
            print(f'Creating torchscript at {params.msg_decoder_path}...')
            torch.jit.save(torchscript_m, params.msg_decoder_path)

    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]

    # 锁定 LDM and hidden decoder中不需要训练的模型参数
    for param in [*msg_decoder.parameters(), *ldm_ae.parameters()]:
        param.requires_grad = False

    # 加载数据
    print(f'加载数据 训练集：{params.train_dir} 测试集：{params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        libs_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size,
                                        num_imgs=params.batch_size * params.steps, shuffle=True, num_workers=4,
                                        collate_fn=None)
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size * 4, num_imgs=1000,
                                      shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([libs_img.unnormalize_vqgan, libs_img.normalize_img])

    # 创建损失函数
    print(f'创建损失函数...')
    print(f'创建损失函数参数: {params.loss_w}  {params.loss_i}...')
    if params.loss_w == 'mse':
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded * temp - (2 * keys - 1)) ** 2)  # b k - b k
    elif params.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded * temp, keys,
                                                                                     reduction='mean')
    else:
        raise NotImplementedError

    if params.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)
    elif params.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif params.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif params.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    else:
        raise NotImplementedError

    for ii_key in range(params.num_keys):

        # 创建模型内的唯一Key
        print(f'\n创建模型内的唯一Key （{nbit} bits）...')
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        key_str = "".join([str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')

        # 复制LDM的参数并且开始微调
        ldm_decoder = deepcopy(ldm_ae)
        ldm_decoder.encoder = nn.Identity()
        ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.to(device)
        for param in ldm_decoder.parameters():
            param.requires_grad = True
        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

        # 训练
        print(f'训练')

        train_stats = train(train_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet,
                            key, params)
        val_stats = val(val_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        log_stats = {'key': key_str,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     }
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

        # Save checkpoint
        torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth"))
        with (Path(params.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
        with (Path(params.output_dir) / "keys.txt").open("a") as f:
            f.write(os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth") + "\t" + key_str + "\n")
        print('\n')


def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable,
          ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet: nn.Module,
          key: torch.Tensor, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        keys = key.repeat(imgs.shape[0], 1)

        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)
        # 图像编码器提取中间层vector
        imgs_z = ldm_ae.encode(imgs)  # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()

        # 分别用原始的decoder和微调过的decoder对中间层vector解码
        imgs_d0 = ldm_ae.decode(imgs_z)  # b z h/f w/f -> b c h w
        imgs_w = ldm_decoder.decode(imgs_z)  # b z h/f w/f -> b c h w

        # 提取水印
        decoded = msg_decoder(vqgan_to_imnet(imgs_w))  # b c h w -> b k

        # 计算原始的decoder和微调过的decoder各自的loss
        lossw = loss_w(decoded, keys)#提取的信息和key计算loss
        lossi = loss_i(imgs_w, imgs_d0)#两个decoder解码后的图像计算loss
        loss = params.lambda_w * lossw + params.lambda_i * lossi

        # 梯度下降
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log记录
        diff = (~torch.logical_xor(decoded > 0, keys > 0))  # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
        word_accs = (bit_accs == 1)  # b
        log_stats = {
            "iteration": ii,
            "loss": loss.item(),
            "loss_w": lossw.item(),
            "loss_i": lossi.item(),
            "psnr": libs_img.psnr(imgs_w, imgs_d0).mean().item(),#峰值信噪比
            # "psnr_ori": libs_img.psnr(imgs_w, imgs).mean().item(),
            "bit_acc_avg": torch.mean(bit_accs).item(),
            "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})
        if ii % params.log_freq == 0:
            print(json.dumps(log_stats))

        # 保存训练图像
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(libs_img.unnormalize_vqgan(imgs), 0, 1),
                       os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
            save_image(torch.clamp(libs_img.unnormalize_vqgan(imgs_d0), 0, 1),
                       os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
            save_image(torch.clamp(libs_img.unnormalize_vqgan(imgs_w), 0, 1),
                       os.path.join(params.imgs_dir, f'{ii:03}_train_w.png'), nrow=8)

    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module,
        vqgan_to_imnet: nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):

        imgs = imgs.to(device)

        imgs_z = ldm_ae.encode(imgs)  # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()

        imgs_d0 = ldm_ae.decode(imgs_z)  # b z h/f w/f -> b c h w
        imgs_w = ldm_decoder.decode(imgs_z)  # b z h/f w/f -> b c h w

        keys = key.repeat(imgs.shape[0], 1)

        log_stats = {
            "iteration": ii,
            "psnr": libs_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": libs_img.psnr(imgs_w, imgs).mean().item(),
        }
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: libs_img.center_crop(x, 0.1),
            'crop_05': lambda x: libs_img.center_crop(x, 0.5),
            'rot_25': lambda x: libs_img.rotate(x, 25),
            'rot_90': lambda x: libs_img.rotate(x, 90),
            'resize_03': lambda x: libs_img.resize(x, 0.3),
            'resize_07': lambda x: libs_img.resize(x, 0.7),
            'brightness_1p5': lambda x: libs_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: libs_img.adjust_brightness(x, 2),
            'jpeg_80': lambda x: libs_img.jpeg_compress(x, 80),
            'jpeg_50': lambda x: libs_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w))
            decoded = msg_decoder(imgs_aug)  # b c h w -> b k
            diff = (~torch.logical_xor(decoded > 0, keys > 0))  # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
            word_accs = (bit_accs == 1)  # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(libs_img.unnormalize_vqgan(imgs), 0, 1),
                       os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(libs_img.unnormalize_vqgan(imgs_d0), 0, 1),
                       os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'), nrow=8)
            save_image(torch.clamp(libs_img.unnormalize_vqgan(imgs_w), 0, 1),
                       os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'), nrow=8)

    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    # 生成或读取参数
    parser = get_parser()
    params = parser.parse_args()

    main(params)

