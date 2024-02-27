import argparse
from calendar import EPOCH
import time
import csv
import datetime
from path import Path
from classic_vo import PinholeCamera, VisualOdometry
from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm
from inverse_warp import pose_vec2mat
from kitti_eval.kitti_odometry import KittiEvalOdom
import models

import custom_transforms
from util import tensor2array, save_checkpoint
from datasets.sequence_folders import SequenceFolder
from datasets.pair_folders import PairFolder
from loss_functions import compute_photo_and_geometry_loss_classical, compute_rotation_consistency_loss, compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='compatible KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=0.9)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-r', '--rotation-consistency-weight', type=float, help='weight for rotation consistency with respect to classical methods', metavar='W', default=0.2)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=1, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', metavar='PATH', default= './checkpoints/resnet18_attention/07-18-15:39/dispnet_3_checkpoint.pth.tar',help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', metavar='PATH', default= './checkpoints/resnet18_attention/07-18-15:39/exp_pose_3_checkpoint.pth.tar', help='path to pre-trained pose model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checkpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def main():

    global best_error, n_iter, device
    torch.cuda.empty_cache()
    args = parser.parse_args()
    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    classic_vo = VisualOdometry(cam)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    else:
        train_set = PairFolder(
            args.data,
            seed=args.seed,
            train=True,
            transform=train_transform
        )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform,
            dataset=args.dataset
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
  
   
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseAttentionNet(18, args.with_pretrain).to(device)

    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseAttentionNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)
        print("*** epoch num: ",epoch)
        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer,epoch, classic_vo)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers )
            pose_errors, pose_error_names = validate_with_gt_pose(args, val_loader, pose_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, classic_vo,output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))
        
        if(pose_errors):
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(pose_error_names, pose_errors))
            logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
            
        for error, name in zip(pose_errors, pose_error_names):
            training_writer.add_scalar(name, error, epoch)

        decisive_error = pose_errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best,epoch)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer,epoch, classic_vo):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3, w4 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight, args.rotation_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i,  (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0        
           
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        # compute output

        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
      
        rotm_classic, rotm_inv_classic = compute_classic_poses_with_inv(classic_vo, tgt_img, ref_imgs)       
      
        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        
        loss_4 = compute_rotation_consistency_loss(rotm_classic, rotm_inv_classic, poses, poses_inv)
        
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4

        if log_losses:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('rotation_consistency_loss', loss_4.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {} epoch {}'.format(batch_time, data_time, losses,epoch))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger,  classic_vo, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=10),
                                        epoch)
            output_writers[i].add_image('val Depth Output 1st scale',
                                        tensor2array(tgt_depth[1][0], max_value=10),
                                        epoch)
            output_writers[i].add_image('val Depth Output 2st scale',
                                        tensor2array(tgt_depth[2][0], max_value=10),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        
        rotm_classic, rotm_inv_classic = compute_classic_poses_with_inv(classic_vo, tgt_img, ref_imgs)
        loss_4 = compute_rotation_consistency_loss(rotm_classic, rotm_inv_classic, poses, poses_inv)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()
        loss_4 = loss_4.item()
        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss', "Rotational_loss"]

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

@torch.no_grad()
def validate_with_gt_pose(args, val_loader, pose_net, epoch, logger, output_writers=[]):
    
    args.dataset_dir = './dataset/sequences/'
    args.sequences = ['09','10']
    
    args.sequence = None
    args.no_resize = None
    args.output_dir = ('{}/results_pose_{}'.format(args.save_path, epoch))
    args.img_exts = ['png', 'jpg', 'bmp']
    args.img_width= 832
    args.img_height = 256
    pose_net.eval()
    
    for i in args.sequences:
        image_dir = Path(args.dataset_dir + i + "/image_2/")
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()

        test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
        test_files.sort()

        print('{} files to test'.format(len(test_files)))

        global_pose = np.eye(4)
        poses = [global_pose[0:3, :].reshape(1, 12)]

        n = len(test_files)
        tensor_img1 = load_tensor_image(test_files[0], args)

        for iter in tqdm(range(n - 1)):

            tensor_img2 = load_tensor_image(test_files[iter+1], args)

            pose = pose_net(tensor_img1, tensor_img2)

            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)

            poses.append(global_pose[0:3, :].reshape(1, 12))
            
            # update
            tensor_img1 = tensor_img2

        poses = np.concatenate(poses, axis=0)
        filename = Path(i + ".txt")
        print(filename, 'asasasasa')
        np.savetxt('{}/{}'.format(output_dir, filename), poses, delimiter=' ', fmt='%1.8e')
    
    eval_tool = KittiEvalOdom()
    gt_dir = "./kitti_eval/gt_poses/"
    result_dir = args.output_dir
    ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot = eval_tool.eval(
        gt_dir,
        result_dir,
        alignment='7dof',
        seqs=args.sequence,
    )
    error_names = ['ave_t_err', 'ave_r_err','ate', 'rpe_trans', 'rpe_rot']
    errors = AverageMeter(i=len(error_names))
    errors.update([ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot])
    return errors.avg, error_names

@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:, 0]

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image('val target Depth',
                                            tensor2array(depth_to_show, max_value=10),
                                            epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                output_writers[i].add_image('val target Disparity Normalized',
                                            tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                            epoch)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(output_depth[0], max_value=10),
                                        epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
  

    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))
     
    return poses, poses_inv

def compute_classic_poses_with_inv(VO, tgt_img, ref_imgs):
    of_rots = []
    of_rots_inv = []
    
    for ref_img in ref_imgs:    
        of_rot = []
        of_rot_inv = []  
        for i in range(len(tgt_img)):
            
            target_image = tgt_img[i].cpu().detach().permute(1, 2, 0).numpy()
            reference_image = ref_img[i].cpu().detach().permute(1, 2, 0).numpy()
            rmat = VO.update(target_image, reference_image)
            rmat_inv = VO.update(reference_image, target_image)            
            of_rot.append(rmat)
            of_rot_inv.append(rmat_inv)
            
    of_rots.append(of_rot)
    of_rots_inv.append(of_rot_inv)    

    return of_rots, of_rots_inv

if __name__ == '__main__':
    main()
