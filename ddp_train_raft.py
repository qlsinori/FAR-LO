# -*- coding:UTF-8 -*-

import os
import sys
import torch
import torch.profiler
# from torch.utils.tensorboard import SummaryWriter
from tools.euler_tools import quat2mat
import datetime
import torch.utils.data
import numpy as np
import time
import pickle
from tqdm import tqdm
import argparse
from ddp_configs_raft import regformer_args

from tools.logger_tools import log_print, creat_logger

from kitti_pytorch import points_dataset
from ddp_regformer_model_raft import regformer_model, get_loss
# from regformer_model_1c_1l import regformer_model, get_loss
from tools.collate_functions_multi_gpu import collate_pair
from easydict import EasyDict
import colorlog as logging
from collections.abc import Iterable
import yaml
from refine.loss import RegistrationLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

args = regformer_args()

def update_args(args, cfg: dict):
    def subdict2edict(iterable_ob):
        for i, element in enumerate(iterable_ob):
            if isinstance(element, dict):
                iterable_ob[i] = EasyDict(element)
            elif isinstance(element, Iterable) and not isinstance(element, str):
                subdict2edict(element)

    for key, value in cfg.items():
        if not hasattr(args, key):
            logger.warning(f'Found unknown parameter in yaml file: {key}')
        if isinstance(value, dict):
            value = EasyDict(value)
        elif isinstance(value, Iterable) and not isinstance(value, str):
            subdict2edict(value)
        setattr(args, key, value)
    return args

'''CREATE DIR'''
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
# experiment dir
experiment_dir = os.path.join(base_dir, 'experiment')
if not args.task_name:
    file_dir = os.path.join(experiment_dir, '{}_KITTI_{}'.format(args.model_name, str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))))
else:
    file_dir = os.path.join(experiment_dir, args.task_name)

    # eval dir
eval_dir = os.path.join(file_dir, 'eval')

# log dir
log_dir = os.path.join(file_dir, 'logs')

tb_log_dir = os.path.join(log_dir, 'logging')


writer = None

checkpoints_dir = os.path.join(file_dir, 'ckpt')

def bkup(local_rank):

    if local_rank == 0:

        if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

        if not os.path.exists(file_dir): os.makedirs(file_dir)

        if not os.path.exists(eval_dir): os.makedirs(eval_dir)

        if not os.path.exists(log_dir): os.makedirs(log_dir)

        if not os.path.exists(checkpoints_dir): os.makedirs(checkpoints_dir)

        if not os.path.exists(tb_log_dir): os.makedirs(tb_log_dir)

        os.system('cp %s %s' % ('ddp_train_raft.py', log_dir))
        os.system('cp %s %s' % ('ddp_configs_raft.py', log_dir))
        os.system('cp %s %s' % ('ddp_regformer_model_raft.py', log_dir))
        os.system('cp %s %s' % ('regformer_model_utils.py', log_dir))
'''LOG'''


def calc_error_np(pred_R, pred_t, gt_R, gt_t):
    tmp = (np.trace(pred_R.transpose().dot(gt_R)) - 1) / 2
    tmp = np.clip(tmp, -1.0, 1.0)
    L_rot = np.arccos(tmp)
    L_rot = 180 * L_rot / np.pi
    L_trans = np.linalg.norm(pred_t - gt_t)
    return L_rot, L_trans


def main():
    global args
    torch.cuda.empty_cache()

    train_dir_list = [0,1,2,3]
    # val_dir_list = [6, 7]
    test_dir_list = [4,5]

    local_rank = args.local_rank

    bkup(local_rank)

    if local_rank == 0:
        logger = creat_logger(log_dir, args.model_name)
        logger.info('----------------------------------------TRAINING----------------------------------')
        logger.info('PARAMETER ...')
        logger.info(args)


    model = regformer_model(args, args.batch_size, args.H_input, args.W_input, args.is_training,
                            show_time=args.show_time, multi_gpu=args.multi_gpu, writer=writer)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo')
    device = torch.device("cuda", local_rank)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    
    model._set_static_graph()
    # train set
    train_dataset = points_dataset(
        is_training=1,
        num_point=args.num_points,
        data_dir_list=train_dir_list,
        config=args,
        data_keep=args.data_keep
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pair,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
        prefetch_factor=4
    )



    print(torch.__version__)


    if args.ckpt is not None:
        file = open(args.ckpt, "rb")

        checkpoint = torch.load(file, map_location=lambda storage, loc: storage.cuda(local_rank))

        pretrained_dict = checkpoint['model_state_dict']


        model_dict = model.state_dict()

        modified_dict = {f"module.{key}": value for key, value in pretrained_dict.items()}

        model_dict.update(modified_dict)

        model.load_state_dict(model_dict, strict=True)


        init_epoch = 0
        if local_rank == 0:
            log_print(logger, 'load model {}'.format(args.ckpt))

    else:
        init_epoch = 0
        if local_rank == 0:
            log_print(logger, 'Training from scratch')


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                    momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize,
                                                gamma=args.lr_gamma, last_epoch=-1)

    train_name =  ['list']

    for name, param in model.named_parameters():
        if any(train in str(name) for train in train_name):
            print('train_name:',name, param.requires_grad)
            param.requires_grad = True
        else:
            # print('not train_name:', name, param.requires_grad)
            param.requires_grad = False

    if args.eval_before == 1 :
        if local_rank == 0:
            eval_pose(model, [4, 4], init_epoch)
        elif dist.get_rank() == 1:
            eval_pose(model, [1, 1], init_epoch)
        elif dist.get_rank() == 2:
            eval_pose(model, [2, 2], init_epoch)
        elif dist.get_rank() == 3:
            eval_pose(model, [3, 3], init_epoch)
        elif dist.get_rank() == 4:
            eval_pose(model, [0, 0], init_epoch)
        elif dist.get_rank() == 5:
            eval_pose(model, [5, 5], init_epoch)

        if local_rank == 0:
            save_path = os.path.join(checkpoints_dir,
                                     '{}_{:03d}_{:04f}.pth.tar'.format(model.__class__.__name__, -1,
                                                                       0))
            torch.save({
                'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
            }, save_path)
            if local_rank == 0:
                log_print(logger, 'Save {}...'.format(model.__class__.__name__))



       
    
    
    for name, param in model.state_dict(keep_vars=True).items():
        # print(name, param.requires_grad)
        if param.requires_grad and local_rank == 0:
            pass
            # print(name, param.requires_grad)




    optimizer.param_groups[0]['initial_lr'] = args.learning_rate

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    for epoch in range(init_epoch, args.max_epoch):
        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())
        train_loader.sampler.set_epoch(epoch)
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            if args.show_time:
                torch.cuda.synchronize()
            start_train_one_batch = time.time()

            pos2, pos1, T_gt, T_trans, T_trans_inv, Tr = data

            if args.show_time:
                torch.cuda.synchronize()
            # print('load_data_time: ', time.time() - start_train_one_batch)

            if args.multi_gpu is not None:
                pos2 = pos2.cuda()
                pos1 = pos1.cuda()
            else:
                pos2 = [b.cuda() for b in pos2]
                pos1 = [b.cuda() for b in pos1]

            T_trans = T_trans.cuda().to(torch.float32)
            T_trans_inv = T_trans_inv.cuda().to(torch.float32)
            T_gt = T_gt.cuda().to(torch.float32)
            model = model.train()

            if args.show_time:
                torch.cuda.synchronize()

            prepare_time = time.time()
            if args.show_time:
                print('load_data_time : ', prepare_time - start_train_one_batch)


            qt_list, q_gt, t_gt, w_x, w_q = model(pos2, pos1, T_gt, T_trans, T_trans_inv)


            loss = get_loss(qt_list, q_gt, t_gt, w_x, w_q)

            if args.show_time:
                torch.cuda.synchronize()
            model_trans_time = time.time()
            if args.show_time:
                print('model_trans_time : ',model_trans_time  - prepare_time)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.show_time:
                torch.cuda.synchronize()
            # print('load_data_time + model_trans_time + forward + back_ward ', time.time() - start_train_one_batch)

            if args.multi_gpu is not None:
                total_loss += loss.mean().cpu().data * args.batch_size
            else:
                total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size
            res_time = time.time()

            # prof.step()
            if args.show_time:
                print('res_time:', res_time - model_trans_time)
        # Adjusting lr
        train_loss = total_loss / total_seen
        if local_rank == 0:
            log_print(logger, 'EPOCH {} train mean loss: {:04f}'.format(epoch, float(train_loss)))

        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], args.learning_rate_clip)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if epoch < 300:
            if epoch % 2 == 0:
                if  dist.get_rank() == 0:
                    save_path = os.path.join(checkpoints_dir,
                                             '{}_{:03d}_{:04f}.pth.tar'.format(model.__class__.__name__, epoch,
                                                                               float(train_loss)))
                    torch.save({
                        'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                        'opt_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch

                    }, save_path)
                    if local_rank == 0:
                        log_print(logger, 'Save {}...'.format(model.__class__.__name__))

                    eval_pose(model, [0,0], epoch)
                elif dist.get_rank() == 1:
                    eval_pose(model, [1,1], epoch)
                elif dist.get_rank() == 2:
                    eval_pose(model, [2,2], epoch)
                elif dist.get_rank() == 3:
                    eval_pose(model, [3,3], epoch)
                elif dist.get_rank() == 4:
                    eval_pose(model, [4,4], epoch)
                elif dist.get_rank() == 5:
                    eval_pose(model, [5,5], epoch)
        else:
            if epoch % 2 == 0 and local_rank == 0:
                save_path = os.path.join(checkpoints_dir,
                                         '{}_{:03d}_{:04f}.pth.tar'.format(model.__class__.__name__, epoch,
                                                                           float(train_loss)))
                torch.save({
                    'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch

                }, save_path)
                if local_rank == 0:
                    log_print(logger, 'Save {}...'.format(model.__class__.__name__))

                eval_pose(model, test_dir_list, epoch)
                # excel_eval.update(eval_dir)







def eval_pose(model, test_list, epoch):
    for i,item in enumerate(test_list):
        test_dataset = points_dataset(
            is_training=0,
            num_point=args.num_points,
            data_dir_list=[item],
            config=args
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_pair,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
            prefetch_factor=4
        )
        line = 0

        total_time = 0

        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            if args.show_time:
                torch.cuda.synchronize()

            start_prepare = time.time()

            pos2, pos1, T_gt, T_trans, T_trans_inv, Tr = data

            if args.show_time:
                torch.cuda.synchronize()
            # print('data_prepare_time: ', time.time() - start_prepare)

            if args.multi_gpu is not None:
                pos2 = pos2.cuda()
                pos1 = pos1.cuda()
            else:
                pos2 = [b.cuda() for b in pos2]
                pos1 = [b.cuda() for b in pos1]

            pos2, pos1, T_gt, T_trans, T_trans_inv, Tr = data

            T_trans = T_trans.cuda().to(torch.float32)
            T_trans_inv = T_trans_inv.cuda().to(torch.float32)
            T_gt = T_gt.cuda().to(torch.float32)

            model = model.eval()

            with torch.no_grad():

                if args.show_time:
                    torch.cuda.synchronize()
                start_time = time.time()

                qt_list, q_gt, t_gt, w_x, w_q = model(pos2, pos1, T_gt, T_trans, T_trans_inv)

                if args.show_time:
                    torch.cuda.synchronize()
                # print('eval_one_time: ', time.time() - start_time)

                if args.show_time:
                    torch.cuda.synchronize()
                total_time += (time.time() - start_time)
                if i % 2 == 0:
                    l0_q = qt_list[3][0].cpu()
                    l0_t = qt_list[3][1].cpu()
                else:
                    l0_q = qt_list[-1][0].cpu()
                    l0_t = qt_list[-1][1].cpu()
                pred_q = l0_q.numpy()
                pred_t = l0_t.numpy()
                # print(pred_q.shape)
                # print(pred_t.shape)
                # deal with a batch_size
                for n0 in range( l0_q.shape[0] ):
                    # print(f"n0: {n0}, Tr shape: {Tr.shape}")

                    cur_Tr = Tr[n0, :, :]

                    qq = pred_q[n0:n0 + 1, :]
                    qq = qq.reshape(4)
                    tt = pred_t[n0:n0 + 1, :]
                    tt = tt.reshape(3, 1)
                    RR = quat2mat(qq)

                    filler = np.array([0.0, 0.0, 0.0, 1.0])
                    filler = np.expand_dims(filler, axis=0)  ##1*4

                    TT = np.concatenate([np.concatenate([RR, tt], axis=-1), filler], axis=0)

                    TT = np.matmul(cur_Tr, TT)
                    TT = np.matmul(TT, np.linalg.inv(cur_Tr))

                    if line == 0:
                        T_final = TT
                        T = T_final[:3, :]
                        T = T.reshape(1, 1, 12)
                        line += 1
                    else:
                        T_final = np.matmul(T_final, TT)
                        T_current = T_final[:3, :]
                        T_current = T_current.reshape(1, 1, 12)
                        T = np.append(T, T_current, axis=0)



        T = T.reshape(-1, 12)

        traj_npy = os.path.join(log_dir, str(item).zfill(5) + '_pred.npy')

        data_dir = os.path.join(eval_dir, 'translonet_' + str(item).zfill(2))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.save(traj_npy, T)

        os.system('cp %s %s' % (traj_npy, data_dir))  ###SAVE THE txt FILE
        os.system('python evaluation.py --result_dir ' + data_dir + ' --eva_seqs ' + str(item).zfill(
            5) + '_pred' + ' --epoch ' + str(epoch))

        txt = os.path.join(log_dir, str(item).zfill(5) + '_pred.txt')
        np.savetxt(txt, T, fmt='%f', delimiter=' ')
        eva_seq_dir = os.path.join(eval_dir, '{}_eval_{}'.format(str(item).zfill(5), str(epoch)))
        os.system('cp %s %s' % (txt, eva_seq_dir))  ###SAVE THE txt FILE
        os.system('cp %s %s' % (data_dir + '/output.txt', log_dir))

    return 0

def val_pose(model, val_list, epoch):
    total_loss = 0
    count = 0
    for item in val_list:
        val_dataset = points_dataset(
            is_training=0,
            num_point=args.num_points,
            data_dir_list=[item],
            config=args,
            data_keep=args.data_keep
        )

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.eval_batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 collate_fn=collate_pair,
                                                 pin_memory=True,
                                                 worker_init_fn=lambda x: np.random.seed(
                                                     (torch.initial_seed()) % (2 ** 32)),
                                                 prefetch_factor=4
                                                 )  # drop_last

        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                pos2, pos1, T_gt, T_trans, T_trans_inv, Tr = data
                pos2 = [b.cuda() for b in pos2]
                pos1 = [b.cuda() for b in pos1]
                T_trans = T_trans.cuda().to(torch.float32)
                T_trans_inv = T_trans_inv.cuda().to(torch.float32)
                T_gt = T_gt.cuda().to(torch.float32)
                model = model.eval()

                l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_ouput, q_gt, t_gt, w_x, w_q = model(pos2, pos1,
                                                                                                        T_gt, T_trans,
                                                                                                        T_trans_inv)
                loss = get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)
                total_loss += loss.item()
                count += args.eval_batch_size
    total_loss = total_loss / count
    return total_loss


    return 0
if __name__ == '__main__':
    main()
