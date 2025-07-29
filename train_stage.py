import random
import torch
import gc
import numpy as np
import time

from tqdm import tqdm


from checkpoint import save_pretrain_ckpt, EarlyStopping, load_best_full_model
from dataset.utils_multigpu import data_to_device
from loss import VarianceMaximizationLoss
from util import evaluate_ml, evaluate_mc, log_info

from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

class bcolors:
    OKGREEN = '\033[92m'  # 绿色
    FAIL = '\033[91m'      # 红色
    ENDC = '\033[0m'       # 结束颜色



def trainer_downstream(model,args,train_dataloader,val_dataloader,test_dataloader,device,loss_func,bert_optimizer=None,bert_scheduler=None,modal_optimizer=None,modal_scheduler=None,ram_optimizer=None,ram_scheduler=None,other_optimizer=None,other_scheduler=None,writer=None, classifier=None, class_optimizer=None, class_scheduler=None):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_dir=args.save_dir)
    log_path = args.log_dir + '/log.out'
    count = 0
    global_step = 0
    for epoch in range(args.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        # @1 Train epoch
        print(f"Epoch {epoch}:")
        start = time.time()
        train_acc, train_auroc, train_auprc, train_f1, train_auroc_micro, train_loss, count, global_step = trainer_downstream_train_epoch(epoch, count, global_step, model,args,train_dataloader,device,loss_func,bert_optimizer,modal_optimizer,ram_optimizer,other_optimizer,bert_scheduler,modal_scheduler,ram_scheduler,other_scheduler,writer,classifier,class_optimizer,class_scheduler,test_dataloader, early_stopping)
        log_info(log_path, 'Train', epoch, train_acc, start=start, auroc=train_auroc, auprc=train_auprc, f1=train_f1, auroc_micro=train_auroc_micro, loss=train_loss, save=True)

        print(f"Val...")
        start = time.time()
        val_acc, val_auroc, val_auprc, val_f1, val_auroc_micro, val_loss, ts2ns_loss, ns2ts_loss, acc_sub, auroc_sub, auprc_sub, f1_sub, auroc_micro_sub, loss_full, ts2ns_full, ns2ts_full, others = trainer_downstream_eval_epoch(
            epoch, model, args,
            val_dataloader,
            device, loss_func, classifier)
        log_info(log_path, 'Test', epoch, val_acc, start=start, auroc=val_auroc, auprc=val_auprc, f1=val_f1,
                 auroc_micro=val_auroc_micro, loss=val_loss, ns2ts_loss=ns2ts_loss, ts2ns_loss=ts2ns_loss,
                 save=True)

        if args.stage == "finetune":
            log_info(log_path, 'Test - Missing Subset', epoch, acc_sub, start=start, auroc=auroc_sub, auprc=auprc_sub, f1=f1_sub, auroc_micro=auroc_micro_sub, loss=loss_full, ns2ts_loss=ns2ts_full, ts2ns_loss=ts2ns_full, save=True)

            if others != None:
                acc_tsmiss, auroc_tsmiss, auprc_tsmiss, f1_tsmiss, auroc_micro_tsmiss, acc_nsmiss, auroc_nsmiss, auprc_nsmiss, f1_nsmiss, auroc_micro_nsmiss = others
                log_info(log_path, 'Test - TS Missing Subset', epoch, acc_tsmiss, start=start, auroc=auroc_tsmiss,
                         auprc=auprc_tsmiss, f1=f1_tsmiss, auroc_micro=auroc_micro_tsmiss, loss=loss_full,
                         ns2ts_loss=ns2ts_full, ts2ns_loss=ts2ns_full, save=True)
                log_info(log_path, 'Test - NS Missing Subset', epoch, acc_nsmiss, start=start, auroc=auroc_nsmiss,
                         auprc=auprc_nsmiss, f1=f1_nsmiss, auroc_micro=auroc_micro_nsmiss, loss=loss_full,
                         ns2ts_loss=ns2ts_full, ts2ns_loss=ts2ns_full, save=True)

        rank = dist.get_rank()
        if rank == 0 and early_stopping is not None:
            if args.stage == "pretrain":
                early_stopping(val_loss, model, classifier, epoch=epoch)
            else:
                if args.eval:
                    # early_stopping(-val_auroc, model, classifier, epoch=epoch)
                    early_stopping(-val_auprc, model, classifier, epoch=epoch)
                else:
                    early_stopping(-val_auroc, model, classifier, epoch=epoch)

            if early_stopping.early_stop:  # and not opt.pretrain:
                print("Early stopping. Training Done.")
                break

        # TODO: val
        # TODO: tired
        # TODO: save model and classifier

    if writer is not None:
        writer.close()
    # TODO: test

    print("Test...")
    # 选择当前最好的模型
    best_epoch = early_stopping.best_epoch

    # 读入训练参数
    checkpoint = load_best_full_model(args, best_epoch, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if classifier is not None:
        model.load_state_dict(checkpoint['classifier'])

    # Test epoch
    start = time.time()
    print(f"Val...")
    start = time.time()
    test_acc, test_auroc, test_auprc, test_f1, test_auroc_micro, test_loss, ts2ns_loss, ns2ts_loss, acc_sub, auroc_sub, auprc_sub, f1_sub, auroc_micro_sub, loss_full, ts2ns_full, ns2ts_full, others = trainer_downstream_eval_epoch(
        epoch, model, args,
        test_dataloader,
        device, loss_func, classifier)
    log_info(log_path, 'Test', epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, f1=test_f1,
             auroc_micro=test_auroc_micro, loss=test_loss, ns2ts_loss=ns2ts_loss, ts2ns_loss=ts2ns_loss, save=True)

    print(f'Test loss: ({test_loss:.6f}).  Test acc: ({test_acc:.6f}).  Test auroc: ({test_auroc:.6f}).  Test auprc: ({test_auprc:.6f}).  Test f1: ({test_f1:.6f}).  Test auroc(micro): ({test_auroc_micro:.6f}).')


def trainer_downstream_train_epoch(epoch, count, global_step, model,args,train_dataloader,device,loss_func,bert_optimizer,modal_optimizer,ram_optimizer,other_optimizer,bert_scheduler=None,modal_scheduler=None,ram_scheduler=None,other_scheduler=None,writer=None,classifier=None, class_optimizer=None, class_scheduler=None, test_dataloader=None, early_stopping=None):
    model.train()
    if classifier is not None:
        classifier.train()

    losses = []
    sup_preds, sup_labels = [], []
    acc, auroc, auprc = 0, 0, 0
    scaler = GradScaler()


    for step, batch in enumerate(tqdm(train_dataloader, position=0)):
        global_step += 1

        # 1. preprocess data
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, ts_miss, ns_miss, label = data_to_device(args.device, batch)

        # 3. input to model
        with autocast():
            out, others = model(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt,
                                 note_tau, note_mask, query_ts_tt=query_tt, query_note_tt=query_tt, query_ts_data=query_ts_data, ns_miss=ns_miss, ts_miss=ts_miss)

            if classifier is not None:
                out = classifier(out)

            # 5. loss
            if args.task == "48ihm":
                label = label.unsqueeze(1)
                loss = torch.sum(loss_func(out, label))
                sup_pred = torch.sigmoid(out)

            if args.stage == "pretrain" and args.baseline_model in ['RAMCare', 'RAMCare_ablation']:
                ts2ns_loss, ns2ts_loss = others
                ts2ns_loss = torch.mean(ts2ns_loss)
                ns2ts_loss = torch.mean(ns2ts_loss)
                # print(f"loss: {loss}    ts2ns: {ts2ns_loss}    ns2ts: {ns2ts_loss}")
                loss = loss + args.lambda_ts2ns * ts2ns_loss + args.lambda_ns2ts * ns2ts_loss
            elif args.stage == "pretrain" and args.baseline_model in ['RAMCare_MIMIC4']:
                ts2ns_loss = others
                ts2ns_loss = torch.mean(ts2ns_loss)
                # print(f"loss: {loss}    ts2ns: {ts2ns_loss}")
                loss = loss + args.lambda_ts2ns * ts2ns_loss

            sup_preds.append(sup_pred.detach().cpu().numpy())
            sup_labels.append(label.detach().cpu().numpy())
            losses.append(loss.item())

            loss = loss / args.gradient_accumulation_steps

        # scaler.scale(loss).backward()
        scaler.scale(loss).backward()

        if torch.isnan(loss).any().item():
            # print grad check # use for check nan
            v_n = []
            v_v = []
            v_g = []
            for name, parameter in model.named_parameters():
                v_n.append(name)
                v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
            for i in range(len(v_n)):
                if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                    color = bcolors.FAIL + '*'
                else:
                    color = bcolors.OKGREEN + ' '
                print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))

        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

            scaler.step(other_optimizer)
            if other_scheduler != None:
                other_scheduler.step()

            if (args.stage == "finetune" and args.data_percent == "all") or (args.stage == "pretrain" and args.baseline_model != "RAMCare_ablation") :
                scaler.step(ram_optimizer)
                if ram_scheduler != None:
                    ram_scheduler.step()
            elif args.baseline_model == 'RAMCare_ablation' and args.wo_module not in ['MOE'] and args.stage == 'pretrain':
                scaler.step(ram_optimizer)
                if ram_scheduler != None:
                    ram_scheduler.step()

            if args.stage == "finetune":
                scaler.step(modal_optimizer)
                if modal_scheduler != None:
                    modal_scheduler.step()

                if bert_optimizer != None:
                    scaler.step(bert_optimizer)
                if bert_scheduler != None:
                    bert_scheduler.step()

            if class_optimizer != None:
                scaler.step(class_optimizer)
            if class_scheduler != None:
                class_scheduler.step()
            scaler.update()

            model.zero_grad()
            if classifier is not None:
                classifier.zero_grad()

        ################    N7    ####################
        if args.rank == 0:
            ##########################################
            if writer is not None:
                writer.add_scalar("loss", loss.item(), global_step)
                if args.stage == "pretrain" and args.baseline_model in ['RAMCare', 'RAMCare_ablation']:
                    writer.add_scalar("ts2ns_loss", ts2ns_loss.item(), global_step)
                    writer.add_scalar("ns2ts_loss", ns2ts_loss.item(), global_step)
                elif args.stage == "pretrain" and args.baseline_model in ['RAMCare_MIMIC4']:
                    writer.add_scalar("ts2ns_loss", ts2ns_loss.item(), global_step)

        del out, loss, sup_pred, label, others
        del name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, ts_miss, ns_miss

        gc.collect()
        torch.cuda.empty_cache()

        # if step == 10:
        #     break
        # finish batch

    # metric
    train_loss = np.average(losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels)
        sup_preds = np.concatenate(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)

        if args.task == '48ihm':
            acc, auroc, auprc, f1, auroc_micro = evaluate_ml(sup_labels, sup_preds)

    # finish epoch
    return acc, auroc, auprc, f1, auroc_micro, train_loss, count, global_step


def trainer_downstream_eval_epoch(epoch, model, args, test_dataloader, device, loss_func, classifier=None, disable=False):
    model.eval()
    if classifier is not None:
        classifier.eval()

    losses = []
    ts2ns_losses = []
    ns2ts_losses = []
    ts2ns_masked_losses = []
    ns2ts_masked_losses = []
    full_masked_losses = []
    sup_preds, sup_labels = [], []
    ts_miss_all, ns_miss_all, full_loss_all = [], [], []

    for step, batch in enumerate(tqdm(test_dataloader, disable=disable)):
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, ts_miss, ns_miss, label = data_to_device(args.device, batch)

        with autocast():
            with torch.no_grad():
                out, others = model(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt,
                                 note_tau, note_mask, query_ts_tt=query_tt, query_note_tt=query_tt, query_ts_data=query_ts_data, ns_miss=ns_miss, ts_miss=ts_miss)

            if classifier is not None:
                out = classifier(out)

            if args.task == "48ihm":
                label = label.unsqueeze(1)
                loss = torch.sum(loss_func(out, label))
                sup_pred = torch.sigmoid(out)

            if args.baseline_model in ['RAMCare', 'RAMCare_ablation']:
                ts2ns_loss, ns2ts_loss = others  # both shape: [B]
            elif args.baseline_model in ['RAMCare_MIMIC4']:
                ts2ns_loss = others
                B = ts2ns_loss.shape[0]
                ns2ts_loss = torch.tensor([0]*B).float().to(ts2ns_loss.device)

            full_mask = (ts_miss == 0) & (ns_miss == 0)
            # print(ts2ns_loss.shape)
            # print(full_mask.shape)
            ts2ns_masked = ts2ns_loss.detach() * full_mask.float()  # full modality ts2ns loss
            ns2ts_masked = ns2ts_loss.detach() * full_mask.float()
            full_masked_loss = loss.detach() * full_mask.float()  # full modality loss

            ts2ns_losses.append(torch.mean(ts2ns_loss).item())
            ns2ts_losses.append(torch.mean(ns2ts_loss).item())
            ts2ns_masked_losses.extend(ts2ns_masked.detach().cpu().tolist())
            ns2ts_masked_losses.extend(ns2ts_masked.detach().cpu().tolist())
            full_masked_losses.extend(full_masked_loss[full_mask].detach().cpu().tolist())

            if args.stage == "pretrain" and args.baseline_model in ['RAMCare', 'RAMCare_ablation']:
                loss = loss + args.lambda_ts2ns * torch.mean(ts2ns_loss) + args.lambda_ns2ts * torch.mean(ns2ts_loss)
            if args.stage == "pretrain" and args.baseline_model in ['RAMCare_MIMIC4']:
                loss = loss + args.lambda_ts2ns * torch.mean(ts2ns_loss)

            sup_preds.append(sup_pred.detach().cpu().numpy())
            sup_labels.append(label.detach().cpu().numpy())
            losses.append(loss.item())
            ts_miss_all.append(ts_miss.detach().cpu())
            ns_miss_all.append(ns_miss.detach().cpu())
            full_loss_all.append(loss.detach().cpu())

        del out, loss, sup_pred, label, others, ts2ns_loss, ns2ts_loss, ts2ns_masked, ns2ts_masked, full_mask, full_masked_loss
        del name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, ts_miss, ns_miss

        gc.collect()
        torch.cuda.empty_cache()

        # if step == 20:
        #     break

    sup_preds = [torch.tensor(pred).to(device) for pred in sup_preds]
    sup_labels = [torch.tensor(label).to(device) for label in sup_labels]
    world_size = dist.get_world_size()
    gathered_preds = gather_tensors(torch.cat(sup_preds), world_size).cpu().numpy()
    gathered_labels = gather_tensors(torch.cat(sup_labels), world_size).cpu().numpy()
    # loss 聚合
    all_losses = gather_tensors(torch.tensor(losses, dtype=torch.float32, device=device), world_size)
    all_ts2ns = gather_tensors(torch.tensor(ts2ns_losses, dtype=torch.float32, device=device), world_size)
    all_ns2ts = gather_tensors(torch.tensor(ns2ts_losses, dtype=torch.float32, device=device), world_size)
    # sub loss 变长tensor聚合
    full_masked_tensor = torch.tensor(full_masked_losses, dtype=torch.float32, device=device)
    ts2ns_masked_tensor = torch.tensor(ts2ns_masked_losses, dtype=torch.float32, device=device)
    ns2ts_masked_tensor = torch.tensor(ns2ts_masked_losses, dtype=torch.float32, device=device)
    all_masked_losses = gather_tensors_variable_length(full_masked_tensor, world_size)
    all_ts2ns_masked = gather_tensors_variable_length(ts2ns_masked_tensor, world_size)
    all_ns2ts_masked = gather_tensors_variable_length(ns2ts_masked_tensor, world_size)

    ts_miss_tensor = torch.cat(ts_miss_all).bool().to(device)
    ns_miss_tensor = torch.cat(ns_miss_all).bool().to(device)
    all_ts_miss = gather_tensors_variable_length(ts_miss_tensor.float(), world_size).bool()
    all_ns_miss = gather_tensors_variable_length(ns_miss_tensor.float(), world_size).bool()
    subset_mask = (all_ts_miss | all_ns_miss).cpu().numpy()
    all_ts_miss = all_ts_miss.cpu().numpy()
    all_ns_miss = all_ns_miss.cpu().numpy()

    rank = dist.get_rank()
    if rank == 0:
        acc, auroc, auprc, f1, auroc_micro = evaluate_and_log(args, gathered_preds, gathered_labels)
        train_loss = all_losses.mean().item()
        ts2ns_loss = all_ts2ns.mean().item()
        ns2ts_loss = all_ns2ts.mean().item()

        if args.stage == "finetune" and args.baseline_model in ['RAMCare', 'RAMCare_ablation']:
            preds_subset = gathered_preds[subset_mask]
            labels_subset = gathered_labels[subset_mask]
            preds_tsmiss = gathered_preds[all_ts_miss]
            labels_tsmiss = gathered_labels[all_ts_miss]
            preds_nsmiss = gathered_preds[all_ns_miss]
            labels_nsmiss = gathered_labels[all_ns_miss]

            acc_sub, auroc_sub, auprc_sub, f1_sub, auroc_micro_sub = evaluate_and_log(args, preds_subset, labels_subset)
            loss_full = all_masked_losses.mean().item()  # full
            ts2ns_full = all_ts2ns_masked.mean().item()  # full
            ns2ts_full = all_ns2ts_masked.mean().item()  # full

            acc_tsmiss, auroc_tsmiss, auprc_tsmiss, f1_tsmiss, auroc_micro_tsmiss = evaluate_and_log(args, preds_tsmiss, labels_tsmiss)
            acc_nsmiss, auroc_nsmiss, auprc_nsmiss, f1_nsmiss, auroc_micro_nsmiss = evaluate_and_log(args, preds_nsmiss, labels_nsmiss)
            others = acc_tsmiss, auroc_tsmiss, auprc_tsmiss, f1_tsmiss, auroc_micro_tsmiss, acc_nsmiss, auroc_nsmiss, auprc_nsmiss, f1_nsmiss, auroc_micro_nsmiss
        elif args.stage == "finetune" and args.baseline_model in ['RAMCare_MIMIC4']:
            preds_subset = gathered_preds[subset_mask]
            labels_subset = gathered_labels[subset_mask]
            preds_nsmiss = gathered_preds[all_ns_miss]
            labels_nsmiss = gathered_labels[all_ns_miss]

            acc_sub, auroc_sub, auprc_sub, f1_sub, auroc_micro_sub = evaluate_and_log(args, preds_subset, labels_subset)
            loss_full = all_masked_losses.mean().item()  # full
            ts2ns_full = all_ts2ns_masked.mean().item()  # full
            ns2ts_full = 0

            acc_tsmiss, auroc_tsmiss, auprc_tsmiss, f1_tsmiss, auroc_micro_tsmiss = 0, 0, 0, 0, 0
            acc_nsmiss, auroc_nsmiss, auprc_nsmiss, f1_nsmiss, auroc_micro_nsmiss = evaluate_and_log(args, preds_nsmiss, labels_nsmiss)
            others = acc_tsmiss, auroc_tsmiss, auprc_tsmiss, f1_tsmiss, auroc_micro_tsmiss, acc_nsmiss, auroc_nsmiss, auprc_nsmiss, f1_nsmiss, auroc_micro_nsmiss
        else:
            acc_sub, auroc_sub, auprc_sub, f1_sub, auroc_micro_sub = 0, 0, 0, 0, 0
            loss_full, ts2ns_full, ns2ts_full = 0, 0, 0
            others = None
    else:
        acc, auroc, auprc, f1, auroc_micro, train_loss, ts2ns_loss, ns2ts_loss = 0, 0, 0, 0, 0, 0, 0, 0
        acc_sub, auroc_sub, auprc_sub, f1_sub, auroc_micro_sub = 0, 0, 0, 0, 0
        loss_full, ts2ns_full, ns2ts_full = 0, 0, 0
        others = None

    return acc, auroc, auprc, f1, auroc_micro, train_loss, ts2ns_loss, ns2ts_loss, acc_sub, auroc_sub, auprc_sub, f1_sub, auroc_micro_sub, loss_full, ts2ns_full, ns2ts_full, others


def gather_tensors(tensor, world_size):
    """
    Gathers tensors from all GPUs and concatenates them on the rank 0 GPU.
    """
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def gather_tensors_variable_length(tensor, world_size):
    local_len = torch.tensor([tensor.shape[0]], device=tensor.device)
    all_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(all_lens, local_len)
    all_lens = [l.item() for l in all_lens]
    max_len = max(all_lens)

    # Pad tensor to max_len
    if tensor.shape[0] < max_len:
        pad_len = max_len - tensor.shape[0]
        padding = torch.zeros(pad_len, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=0)

    # Now gather
    gather_list = [torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)

    # Truncate using original lengths
    gathered = []
    for g, l in zip(gather_list, all_lens):
        gathered.append(g[:l])
    return torch.cat(gathered, dim=0)


def evaluate_and_log(args, sup_preds, sup_labels):
    if len(sup_preds) > 0:
        sup_labels = np.vstack(sup_labels)
        sup_preds = np.vstack(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)

        if args.task == '48ihm' or args.task == "physio":
            acc, auroc, auprc, f1, auroc_micro = evaluate_ml(sup_labels, sup_preds)
        elif args.task == '24pheno':
            print("24pheno!!!!!!!!!!!!")
            n_classes = 25
            acc, auroc, auprc, f1, auroc_micro = evaluate_mc(sup_labels, sup_preds, args.task)

        return acc, auroc, auprc, f1, auroc_micro
    return None, None, None, None, None