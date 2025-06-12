import argparse
import math
import os
import random
import copy
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from engine import *
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, save_ckpt, selective_reinitialize
from tsne import visualize
from collect_feature import collect_feature
from visualize_feature_map import *
def get_args_parser(parser):
    # Model Settings
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--pos_encoding', default='sine', type=str)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--with_box_refine', default=False, type=bool)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj', default=2e-5, type=float)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.5, type=float, help='gradient clipping max norm')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_lr_drop', default=40, type=int)
    # Loss coefficients
    parser.add_argument('--teach_box_loss', default=False, type=bool)
    parser.add_argument('--coef_class', default=2.0, type=float)
    parser.add_argument('--coef_boxes', default=5.0, type=float)
    parser.add_argument('--coef_giou', default=2.0, type=float)
    parser.add_argument('--coef_target', default=1.0, type=float)
    parser.add_argument('--coef_domain', default=1.0, type=float)
    parser.add_argument('--coef_domain_bac', default=0.3, type=float)
    parser.add_argument('--coef_mae', default=1.0, type=float)
    parser.add_argument('--alpha_focal', default=0.25, type=float)
    parser.add_argument('--alpha_ema', default=0.9996, type=float)
    # Dataset parameters
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--source_dataset', default='cityscapes', type=str)
    parser.add_argument('--target_dataset', default='foggy_cityscapes', type=str)
    # Retraining parameters
    parser.add_argument('--epoch_retrain', default=40, type=int)
    parser.add_argument('--keep_modules', default=["decoder"], type=str, nargs="+")
    # MAE parameters
    parser.add_argument('--mae_layers', default=[2], type=int, nargs="+")
    parser.add_argument('--mask_ratio', default=0.3, type=float)
    parser.add_argument('--mr_step', default=0.01, type=float)
    parser.add_argument('--epoch_mae_decay', default=10, type=float)
    # Dynamic threshold (DT) parameters
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--alpha_dt', default=0.5, type=float)
    parser.add_argument('--gamma_dt', default=0.9, type=float)
    parser.add_argument('--max_dt', default=0.45, type=float)
    # mode settings
    parser.add_argument("--mode", default="single_domain", type=str,
                        help="'single_domain' for single domain training, "
                             "'cross_domain_mae' for cross domain training with mae, "
                             "'teaching' for teaching process, 'eval' for evaluation only.")
    # Other settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_loss(epoch, prefix, total_loss, loss_dict):
    writer.add_scalar(prefix + '/total_loss', total_loss, epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(prefix + '/' + k, v, epoch)


def write_ap50(epoch, prefix, m_ap, ap_per_class, idx_to_class):
    writer.add_scalar(prefix + '/mAP50', m_ap, epoch)
    for idx, num in zip(idx_to_class.keys(), ap_per_class):
        writer.add_scalar(prefix + '/AP50_%s' % (idx_to_class[idx]['name']), num, epoch)


def single_domain_training(model, device):
    # Record the start time
    start_time = time.time()
    # Build dataloaders
    train_loader = build_dataloader(args, args.source_dataset, 'source', 'train', train_trans)
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Prepare model for optimization
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    criterion = build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Record the best mAP
    ap50_best = -1.0
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        loss_train, loss_train_dict = train_one_epoch_standard(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        write_loss(epoch, 'single_domain', loss_train, loss_train_dict)
        lr_scheduler.step()
        # Evaluate
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            save_ckpt(model, output_dir / 'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model, output_dir / 'model_last.pth', args.distributed)
        # Write the evaluation results to tensorboard
        write_ap50(epoch, 'single_domain', map50, ap50_per_class, idx_to_class)
    # Record the end time
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Single-domain training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)


def sigmoid_adjustment_step(epoch, total_epochs, base_step=0.2, min_step=0.1, k=10, midpoint=0.5):


    x = epoch / total_epochs

    sigmoid = 1 / (1 + math.exp(-k * (x - midpoint)))

    mr_step = min_step + (base_step - min_step) * (1 - sigmoid)
    return max(mr_step, min_step)


def dynamic_mask_ratio_sigmoid(mask_ratio, loss_list, loss_current, mr_step):


    if len(loss_list) >= 3:
        current_mean = sum(loss_list[-3:]) / 3
    else:
        current_mean = loss_current

    if loss_current >= current_mean:
        mask_ratio -= mr_step
    else:
        mask_ratio += mr_step

    return min(max(mask_ratio, 0.5), 0.8)





def cross_domain_mae(model, device):
    start_time = time.time()
    # Build dataloaders
    source_loader = build_dataloader(args, args.source_dataset, 'source', 'train', strong_trans)
    target_loader = build_dataloader(args, args.target_dataset, 'target', 'train', strong_trans)
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Build MAE branch
    image_size = target_loader.dataset.__getitem__(0)[0].shape[-2:]

    model.transformer.build_mae_decoder(image_size, args.mae_layers, device, channel0=model.backbone.num_channels[0])
    # Prepare model for optimization
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    criterion, criterion_mae = build_criterion(args, device), build_criterion(args, device)
    # optimizer = build_optimizer(args, model)
    optimizer, optimizer_mr = build_optimizer_mae(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Record the best mAP
    ap50_best = -1.0
    mask_ratio = args.mask_ratio
    loss_dict = []
    base_step = 0.1
    min_step = 0.02
    k = 10  # Sigmoid的陡峭程度
    midpoint = 0.5  # Sigmoid的中点
    total_epochs = args.epoch
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(source_loader.sampler, 'set_epoch'):
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        loss_train, loss_train_dict = train_one_epoch_with_mae(
            model=model,
            criterion=criterion,
            criterion_mae=criterion_mae,
            source_loader=source_loader,
            target_loader=target_loader,
            coef_target=args.coef_target,
            # mask_ratio=args.mask_ratio,
            mask_ratio=mask_ratio,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )

        print("epoch", epoch, "mask ratio", mask_ratio, "loss", loss_train, "step", args.mr_step)
        loss_dict.append(loss_train.item())
        if epoch > 3:
 
            mr_step = sigmoid_adjustment_step(epoch, total_epochs, base_step=base_step, min_step=min_step, k=k,
                                              midpoint=midpoint)
            print(f"Epoch {epoch}: Adjusted step = {mr_step}")
        
            mask_ratio = dynamic_mask_ratio_sigmoid(mask_ratio, loss_dict, loss_train.item(), mr_step)
       
            mask_ratio = min(max(mask_ratio, 0), 0.9)
        
 

        write_loss(epoch, 'cross_domain_mae', loss_train, loss_train_dict)
        lr_scheduler.step()
        # Evaluate
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )

        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.0001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            save_ckpt(model, output_dir / 'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model, output_dir / 'model_last.pth', args.distributed)
        # Write the evaluation results to tensorboard
        write_ap50(epoch, 'cross_domain_mae', map50, ap50_per_class, idx_to_class)
    # Record the end time
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Cross-domain MAE training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)



# Teaching
def teaching(model_stu, device):
    start_time = time.time()
    # Build dataloaders
    source_loader = build_dataloader(args, args.source_dataset, 'source', 'train', strong_trans)
    target_loader = build_dataloader_teaching(args, args.target_dataset, 'target', 'train')
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Build teacher model
    model_tch = build_teacher(args, model_stu, device)
    # Build discriminators
    model_stu.build_discriminators(device)
    # Build MAE branch
    image_size = target_loader.dataset.__getitem__(0)[0].shape[-2:]
    model_stu.transformer.build_mae_decoder(image_size, args.mae_layers, device, model_stu.backbone.num_channels[0])
    # Prepare model for optimization
    if args.distributed:
        model_stu = DistributedDataParallel(model_stu, device_ids=[args.gpu], find_unused_parameters=True)
        model_tch = DistributedDataParallel(model_tch, device_ids=[args.gpu])
    criterion = build_criterion(args, device)
    criterion_pseudo = build_criterion(args, device, box_loss=args.teach_box_loss)
    optimizer = build_optimizer(args, model_stu)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Reinitialize checkpoint for selective retraininggpu
    reinit_ckpt = copy.deepcopy(model_tch.state_dict())
    # Initialize thresholds
    thresholds = [args.threshold] * args.num_classes
    # Record the best mAP
    ap50_best = -1.0
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(source_loader.sampler, 'set_epoch'):
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
        loss_train, loss_source_dict, loss_target_dict = train_one_epoch_teaching(
            student_model=model_stu,
            teacher_model=model_tch,
            criterion=criterion,
            criterion_pseudo=criterion_pseudo,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            thresholds=thresholds,
            coef_target=args.coef_target,
            mask_ratio=args.mask_ratio,
            alpha_ema=args.alpha_ema,
            device=device,
            epoch=epoch,
            enable_mae=(epoch < args.epoch_mae_decay),
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Renew thresholds
        alpha_dt = 0.5
        beta = 0.2
        max_dt = 0.45
        min_dt = 0.25
        iterations_per_epoch = 20  # Total iterations per epoch
        total_iters = iterations_per_epoch * args.epoch
        for t in range(1, iterations_per_epoch + 1):
  
            logits_means, logits_variance = criterion.dynamic_threshold2()
            current_iter = (epoch - 1) * iterations_per_epoch + t
    
            gamma = 1 / (1 + math.exp(-alpha_dt * current_iter / total_iters))

       
            new_thresholds = [
                gamma * old + (1 - gamma) * (alpha_dt * math.sqrt(mean) - beta * var)
                for old, mean, var in zip(thresholds, logits_means, logits_variance)
            ]

   
            N = [max(min(threshold, max_dt), min_dt) for threshold in new_thresholds]
            print(f"Dynamic Thresholds at iter {current_iter}: {N}")

        thresholds = N
        # thresholds = criterion.dynamic_threshold(thresholds)
        criterion.clear_positive_logits()
        # Write the losses to tensorboard
        write_loss(epoch, 'teaching_source', loss_train, loss_source_dict)
        write_loss(epoch, 'teaching_target', loss_train, loss_target_dict)
        lr_scheduler.step()
        # Selective Retraining
        if (epoch + 1) % args.epoch_retrain == 0:
            model_stu = selective_reinitialize(model_stu, reinit_ckpt, args.keep_modules)
        # Evaluate teacher and student model
        ap50_per_class_teacher, loss_val_teacher = evaluate(
            model=model_tch,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        ap50_per_class_student, loss_val_student = evaluate(
            model=model_stu,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Save the best checkpoint
        map50_tch = np.asarray([ap for ap in ap50_per_class_teacher if ap > -0.001]).mean().tolist()
        map50_stu = np.asarray([ap for ap in ap50_per_class_student if ap > -0.001]).mean().tolist()
        write_ap50(epoch, 'teaching_teacher', map50_tch, ap50_per_class_teacher, idx_to_class)
        write_ap50(epoch, 'teaching_student', map50_stu, ap50_per_class_student, idx_to_class)
        if max(map50_tch, map50_stu) > ap50_best:
            ap50_best = max(map50_tch, map50_stu)
            save_ckpt(model_tch if map50_tch > map50_stu else model_stu, output_dir / 'model_best.pth',
                      args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model_tch, output_dir / 'model_last_tch.pth', args.distributed)
            save_ckpt(model_stu, output_dir / 'model_last_stu.pth', args.distributed)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching finished. Time cost: ' + total_time_str + ' . Best mAP50: ' + str(ap50_best), flush=args.flush)


# Evaluate only
def eval_only(model, device):
    if args.distributed:
        Warning('Evaluation with distributed mode may cause error in output result labels.')
    criterion = build_criterion(args, device)
    # Eval source or target dataset
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'test', val_trans)
    ap50_per_class, epoch_loss_val, coco_data = evaluate(
        model=model,
        criterion=criterion,
        data_loader_val=val_loader,
        output_result_labels=True,
        device=device,
        print_freq=args.print_freq,
        flush=args.flush
    )
    print('Evaluation finished. mAPs: ' + str(ap50_per_class) + '. Evaluation loss: ' + str(epoch_loss_val))
    output_file = output_dir / 'evaluation_result_labels.json'
    print("Writing evaluation result labels to " + str(output_file))
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(coco_data, fp)


def analysis_only(model, device):

    print("Starting analysis phase...")
    source_loader = build_dataloader(args, args.source_dataset, 'source', 'train', val_trans)
    target_loader = build_dataloader(args, args.target_dataset, 'target', 'train', val_trans)

    # Assume that the model is in evaluation mode
    model.eval()

    # Load data for analysis (you can use your own data loader here)
    # For example, you can use `train_source_loader` and `train_target_loader` to load source and target data respectively
    # Assuming DataLoader and feature_extractor are already defined
    source_feature = collect_feature(source_loader, model, device)
    target_feature = collect_feature(target_loader, model, device)

    # Visualize the features with t-SNE (assuming `visualize` is a separate function for t-SNE plotting)
    print("Visualizing source and target domain features with t-SNE...")
    # 使用 t-SNE 可视化
    visualize(
        source_feature=source_feature,
        target_feature=target_feature,
        filename='tsne_original_features.png',
        source_color='r',
        target_color='b'
    )


    # Perform additional analysis, e.g., evaluate performance metrics or feature comparisons
    # Example: print feature statistics or evaluate the model performance on a specific task

    print("Analysis complete.")


def vis(model, device):
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'test', val_trans)

    if hasattr(val_loader.dataset, 'coco') or hasattr(val_loader.dataset, 'anno_file'):
        evaluator = CocoEvaluator(val_loader.dataset.coco)
        coco_data = json.load(open(val_loader.dataset.anno_file, 'r'))
        dataset_annotations = {image['id']: [] for image in coco_data['images']}
    else:
        raise ValueError('Unsupported dataset type.')

    # img_dir = r"F:\Deraining\our_try\vis_add_Images\image/"
    save_dir = r"./ablation_study/all/"  # 保存图像的文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()

    for i, (images, masks, annotations) in enumerate(val_loader):
        # 获取 image_id
        image_id = annotations[0]['image_id'].item()  # 取 batch 内的第一个 image_id
        image_name = f"{image_id}.png"  # 生成文件名

        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        ori_size = images.size()[2:]
        print(f"Processing Image ID: {image_id}, Original Size: {ori_size}")

        with torch.no_grad():
            conv_img = model(images, masks)
            conv_img = conv_img['features']
            conv_img = torch.nn.functional.interpolate(conv_img, size=ori_size, mode='bilinear', align_corners=False)
            print(conv_img.size())

        # conv_img = torch.nn.functional.interpolate(conv_img, size=ori_size, mode='bilinear', align_corners=False)
        # normalized_conv_img = normalize_feature_map(conv_img)

        # 使用 image_id 作为文件名
        visualize_feature_map(conv_img, save_dir, image_name, selected_channels=None)


def main():
    # Initialize distributed mode
    init_distributed_mode(args)
    # Set random seed
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_random_seed(args.random_seed + get_rank())
    # Print args
    print('-------------------------------------', flush=args.flush)
    print('Logs will be written to ' + str(logs_dir))
    print('Checkpoints will be saved to ' + str(output_dir))
    print('-------------------------------------', flush=args.flush)
    for key, value in args.__dict__.items():
        print(key, value, flush=args.flush)
    # Build model
    device = torch.device(args.device)
    model = build_model(args, device)

    if args.resume != "":
        model = resume_and_load(model, args.resume, device)
    # Training or evaluation
    print('-------------------------------------', flush=args.flush)
    if args.mode == "single_domain":
        single_domain_training(model, device)
    elif args.mode == "cross_domain_mae":
        cross_domain_mae(model, device)
    elif args.mode == "teaching":
        teaching(model, device)
    elif args.mode == "eval":
        eval_only(model, device)
    elif args.mode == "analysis":
        analysis_only(model, device)
    elif args.mode == "vis":
        vis(model, device)
    else:
        raise ValueError('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    # Parse arguments
    parser_main = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    get_args_parser(parser_main)
    args = parser_main.parse_args()
    # Set output directory
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / 'data_logs'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    # Call main function
    main()
