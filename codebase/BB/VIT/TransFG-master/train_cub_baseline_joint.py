# coding=utf-8
# sync test~1028 0439
from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append(os.path.abspath("root-path"))

import argparse
import logging
import os
import random
import time
import warnings
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.modeling import CONFIGS, VisionTransformer_baseline
from models.Gated_Logic_Net import Gated_Logic_Net
from utils.data_utils import get_loader
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def get_correct_multi_label(y_hat, y):
    y_hat = y_hat > 0.5
    correct = y.eq(y_hat).sum().item()
    return correct


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def save_model(args, model, chk_pt_path, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(chk_pt_path, f"{args.name}_{global_step}_checkpoint.bin")
    """
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    """
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    torch.save(checkpoint, os.path.join(chk_pt_path, f"{args.name}_best_checkpoint.bin"))
    logger.info("Saved model checkpoint to [DIR: %s]", chk_pt_path)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "cub":
        num_classes = 200
    elif args.dataset == "DLCV_1":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    model = VisionTransformer_baseline(config, args.img_size, zero_head=True, num_classes=num_classes,
                                       op_size=len(args.concept_names),
                                       smoothing_value=args.smoothing_value)

    checkpoint = np.load(args.pretrained_dir)
    model.load_from(checkpoint)
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def valid(args, criterion_c, criterion_g, model, g, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    correct = 0
    for step, batch in enumerate(epoch_iterator):
        x, label, attribute = batch
        with torch.no_grad():
            x = x.to(args.device)
            label = label.to(torch.long).to(args.device)
            attribute = attribute.to(args.device, dtype=torch.float)
            logits = model(x)
            y_hat, _, _ = g(torch.sigmoid(logits))

            loss_c = criterion_c(logits, attribute)
            loss_g = criterion_g(y_hat, label)
            eval_loss = loss_g + args.lambda_joint * loss_c
            eval_losses.update(eval_loss.item())
            preds = torch.sigmoid(logits)

            correct += y_hat.argmax(dim=1).eq(label).sum().item()

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    val_accuracy = (correct / (
            len(test_loader.dataset) * 108))
    # accuracy = torch.tensor(accuracy).to(args.device)
    # # dist.barrier()
    # val_accuracy = accuracy
    # val_accuracy = val_accuracy.detach().cpu().numpy()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    print(f"step: {global_step} || val_acc: {val_accuracy * 100}")
    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy * 100, global_step=global_step)

    return val_accuracy


def train(args, model, chk_pt_path, output_path, tb_logs_path, seed):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(tb_logs_path, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    g = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(args.device)
    optimizer_c = torch.optim.SGD(model.parameters(),
                                  lr=args.learning_rate,
                                  momentum=0.9,
                                  weight_decay=args.weight_decay)
    optimizer_g = torch.optim.SGD(g.explainer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer_c, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer_c, warmup_steps=args.warmup_steps, t_total=t_total)
    """
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
    """

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    g.zero_grad()
    criterion_c = torch.nn.BCEWithLogitsLoss()
    criterion_g = torch.nn.CrossEntropyLoss()
    set_seed(seed, args.n_gpu)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    while True:
        model.train()
        g.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        correct = 0
        for step, batch in enumerate(epoch_iterator):
            x, label, attribute = batch
            label = label.to(torch.long).to(args.device)
            x = x.to(args.device)
            attribute = attribute.to(args.device, dtype=torch.float)
            logits = model(x)
            y_hat, _, _ = g(torch.sigmoid(logits))

            loss_c = criterion_c(logits, attribute)
            loss_g = criterion_g(y_hat, label)
            loss = loss_g + args.lambda_joint * loss_c

            preds = torch.sigmoid(logits)
            correct += y_hat.argmax(dim=1).eq(label).sum().item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            """
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            """
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                """
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                """
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer_c.step()
                optimizer_c.zero_grad()
                optimizer_g.step()
                optimizer_g.zero_grad()

                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    with torch.no_grad():
                        accuracy = valid(
                            args, criterion_c, criterion_g, model, g, writer, test_loader, global_step
                        )
                    if args.local_rank in [-1, 0]:
                        if best_acc < accuracy:
                            print("saving model!!")
                            save_model(args, model, chk_pt_path, global_step)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break
        train_accuracy = (correct / (
                len(train_loader.dataset) * 108)) * 100
        # accuracy = simple_accuracy(all_preds, all_label)
        # accuracy = torch.tensor(accuracy).to(args.device)
        # dist.barrier()
        # train_accuracy = accuracy
        # train_accuracy = train_accuracy.cpu().numpy()
        logger.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--data-root', metavar='DIR',
                        default='root-path/data/CUB_200_2011',
                        help='path to dataset')
    parser.add_argument('--json-root', metavar='DIR',
                        default='root-path/codebase/data_preprocessing',
                        help='path to json files containing train-val-test split')
    parser.add_argument('--logs', metavar='DIR',
                        default='root-path/log',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='root-path/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='root-path/out',
                        help='path to output logs')
    parser.add_argument('--attribute-file-name', metavar='file',
                        default='attributes.npy',
                        help='file containing all the concept attributes')
    parser.add_argument("--name", default="VIT_CUBS",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "DLCV_1"],
                        default="cub",
                        help="Which dataset.")

    # parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/DL_CV/HW1/training_data/')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument(
        "--pretrained_dir", type=str,
        default="root-path/checkpoints/pretrained_VIT/ViT-B_16.npz",
        help="Where to search for pretrained ViT models."
    )
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    # parser.add_argument("--output_dir", default="./output", type=str,
    #                     help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="100Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    """
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    """

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--temperature-lens', default=0.7, type=float, help='temperature for entropy layer in lens')

    parser.add_argument('--lambda_joint', default=0.01, type=float, help='cem or cbm')
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument('--conceptizator', default='identity_bool', type=str, help='activation')
    parser.add_argument('--concept-names', nargs='+',
                        default=['has_bill_shape_dagger', 'has_bill_shape_hooked_seabird',
                                 'has_bill_shape_allpurpose', 'has_bill_shape_cone', 'has_wing_color_brown',
                                 'has_wing_color_grey', 'has_wing_color_yellow', 'has_wing_color_black',
                                 'has_wing_color_white', 'has_wing_color_buff', 'has_upperparts_color_brown',
                                 'has_upperparts_color_grey', 'has_upperparts_color_yellow',
                                 'has_upperparts_color_black', 'has_upperparts_color_white',
                                 'has_upperparts_color_buff', 'has_underparts_color_brown',
                                 'has_underparts_color_grey', 'has_underparts_color_yellow',
                                 'has_underparts_color_black', 'has_underparts_color_white',
                                 'has_underparts_color_buff', 'has_breast_pattern_solid',
                                 'has_breast_pattern_striped', 'has_breast_pattern_multicolored',
                                 'has_back_color_brown', 'has_back_color_grey', 'has_back_color_yellow',
                                 'has_back_color_black', 'has_back_color_white', 'has_back_color_buff',
                                 'has_tail_shape_notched_tail', 'has_upper_tail_color_brown',
                                 'has_upper_tail_color_grey', 'has_upper_tail_color_black',
                                 'has_upper_tail_color_white', 'has_upper_tail_color_buff',
                                 'has_head_pattern_plain', 'has_head_pattern_capped',
                                 'has_breast_color_brown', 'has_breast_color_grey',
                                 'has_breast_color_yellow', 'has_breast_color_black',
                                 'has_breast_color_white', 'has_breast_color_buff', 'has_throat_color_grey',
                                 'has_throat_color_yellow', 'has_throat_color_black',
                                 'has_throat_color_white', 'has_eye_color_black',
                                 'has_bill_length_about_the_same_as_head',
                                 'has_bill_length_shorter_than_head', 'has_forehead_color_blue',
                                 'has_forehead_color_brown', 'has_forehead_color_grey',
                                 'has_forehead_color_yellow', 'has_forehead_color_black',
                                 'has_forehead_color_white', 'has_forehead_color_red',
                                 'has_under_tail_color_brown', 'has_under_tail_color_grey',
                                 'has_under_tail_color_yellow', 'has_under_tail_color_black',
                                 'has_under_tail_color_white', 'has_under_tail_color_buff',
                                 'has_nape_color_blue', 'has_nape_color_brown', 'has_nape_color_grey',
                                 'has_nape_color_yellow', 'has_nape_color_black', 'has_nape_color_white',
                                 'has_nape_color_buff', 'has_belly_color_grey', 'has_belly_color_yellow',
                                 'has_belly_color_black', 'has_belly_color_white', 'has_belly_color_buff',
                                 'has_wing_shape_roundedwings', 'has_size_small_5__9_in',
                                 'has_size_medium_9__16_in', 'has_size_very_small_3__5_in',
                                 'has_shape_perchinglike', 'has_back_pattern_solid',
                                 'has_back_pattern_striped', 'has_back_pattern_multicolored',
                                 'has_tail_pattern_solid', 'has_tail_pattern_multicolored',
                                 'has_belly_pattern_solid', 'has_primary_color_brown',
                                 'has_primary_color_grey', 'has_primary_color_yellow',
                                 'has_primary_color_black', 'has_primary_color_white',
                                 'has_primary_color_buff', 'has_leg_color_grey', 'has_leg_color_black',
                                 'has_leg_color_buff', 'has_bill_color_grey', 'has_bill_color_black',
                                 'has_crown_color_blue', 'has_crown_color_brown', 'has_crown_color_grey',
                                 'has_crown_color_yellow', 'has_crown_color_black', 'has_crown_color_white',
                                 'has_wing_pattern_solid', 'has_wing_pattern_striped',
                                 'has_wing_pattern_multicolored'])

    parser.add_argument('--labels', nargs='+',
                        default=['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani',
                                 'Crested_Auklet',
                                 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird',
                                 'Red_winged_Blackbird',
                                 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting',
                                 'Lazuli_Bunting',
                                 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird',
                                 'Yellow_breasted_Chat',
                                 'Eastern_Towhee',
                                 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant',
                                 'Bronzed_Cowbird',
                                 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo',
                                 'Mangrove_Cuckoo',
                                 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker',
                                 'Acadian_Flycatcher',
                                 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher',
                                 'Scissor_tailed_Flycatcher',
                                 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar',
                                 'Gadwall',
                                 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe',
                                 'Horned_Grebe',
                                 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak',
                                 'Pine_Grosbeak',
                                 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull',
                                 'Glaucous_winged_Gull',
                                 'Heermann_Gull',
                                 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull',
                                 'Anna_Hummingbird',
                                 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear',
                                 'Long_tailed_Jaeger',
                                 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco',
                                 'Tropical_Kingbird',
                                 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher',
                                 'Ringed_Kingfisher',
                                 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon',
                                 'Mallard',
                                 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird',
                                 'Nighthawk',
                                 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole',
                                 'Orchard_Oriole',
                                 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee',
                                 'Sayornis',
                                 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven',
                                 'White_necked_Raven',
                                 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike',
                                 'Baird_Sparrow',
                                 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow',
                                 'House_Sparrow',
                                 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow',
                                 'Henslow_Sparrow',
                                 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow',
                                 'Savannah_Sparrow',
                                 'Seaside_Sparrow',
                                 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow',
                                 'White_throated_Sparrow',
                                 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow',
                                 'Tree_Swallow',
                                 'Scarlet_Tanager',
                                 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern',
                                 'Elegant_Tern', 'Forsters_Tern',
                                 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher',
                                 'Black_capped_Vireo',
                                 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo',
                                 'White_eyed_Vireo',
                                 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler',
                                 'Black_throated_Blue_Warbler',
                                 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler',
                                 'Chestnut_sided_Warbler',
                                 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler',
                                 'Mourning_Warbler',
                                 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler',
                                 'Pine_Warbler',
                                 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler',
                                 'Wilson_Warbler',
                                 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush',
                                 'Louisiana_Waterthrush',
                                 'Bohemian_Waxwing',
                                 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker',
                                 'Red_bellied_Woodpecker',
                                 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren',
                                 'Cactus_Wren',
                                 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren',
                                 'Common_Yellowthroat'])

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    args.arch = args.model_type
    args.epochs = 95
    args.lr = args.learning_rate

    for seed in range(args.seed):
        print("==============================================")
        print(f"seed: {seed}")
        args.train_batch_size = 16
        args.eval_batch_size = 16
        args.input_size_pi = 2048
        args.hidden_nodes = [10]
        print(f"Batch size: {args.train_batch_size} , {args.eval_batch_size}")
        print("==============================================")
        root = f"lr_{args.lr}_epochs_{args.num_steps}"
        chk_pt_path = os.path.join(
            args.checkpoints, args.dataset, "Baseline-joint-cbm", f"seed_{seed}", args.arch, "explainer"
        )
        output_path = os.path.join(
            args.output, args.dataset, "Baseline-joint-cbm", f"seed_{seed}", args.arch, "explainer"
        )
        tb_logs_path = os.path.join(
            args.logs, args.dataset, "Baseline-joint-cbm", f"seed_{seed}", f"{root}_{args.arch}", "explainer"
        )
        os.makedirs(chk_pt_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(tb_logs_path, exist_ok=True)

        print(chk_pt_path)
        print(output_path)
        print(tb_logs_path)

        os.makedirs(chk_pt_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(tb_logs_path, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            filename=os.path.join(f"{output_path}", "VIT.log"),
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
        )
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                       (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

        # Set seed
        set_seed(seed, args.n_gpu)

        # Model & Tokenizer Setup
        args, model = setup(args)
        # Training
        train(args, model, chk_pt_path, output_path, tb_logs_path, seed)


if __name__ == "__main__":
    main()
