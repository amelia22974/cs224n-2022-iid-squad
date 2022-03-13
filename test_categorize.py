"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, BiDAFSelfAttended, BiDAFSelfAttendedOld, BiDAFCoattended, BiDAFCombined
from models_charemb import BiDAFChar #import BiDAF with Character Embeddings
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD, eval_dict_categorized

def main(args):
    # Set up logging
    type = "charembed"
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # # Get model
    if type == "baseline":

        log.info('Building model...')
        model = BiDAF(word_vectors=word_vectors,
                       hidden_size=args.hidden_size)
        model = nn.DataParallel(model, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()
    elif type == "selfmatchcombined":
        # Get embeddings [CHARACTER EMBEDDINGS VERSION]
        log.info('Loading embeddings...')
        hidden_size = 100
        #drop_prob = 0.2
        use_char_emb = True
        word_vectors = util.torch_from_json(args.word_emb_file)
        char_vectors = util.torch_from_json(args.char_emb_file)
        # Get model
        log.info('Building model...')
        #model = BiDAFCombined(word_vectors=word_vectors,
        #            char_vectors=char_vectors,
        #            hidden_size=hidden_size,
        #            use_char_emb=use_char_emb)

        model = BiDAFCombined(word_vectors=word_vectors,
                    char_vectors=char_vectors,
                    hidden_size=args.hidden_size,
                    drop_prob=0,
                    char_drop_prob=0,
                    use_char_emb=True) #confirm that we're actually using character embeddings
        model = nn.DataParallel(model, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()
    elif type == "selfmatchattention":
        # Get embeddings [CHARACTER EMBEDDINGS VERSION]
        log.info('Loading embeddings...')
        hidden_size = 100
        #drop_prob = 0.2
        use_char_emb = True
        word_vectors = util.torch_from_json(args.word_emb_file)
        char_vectors = util.torch_from_json(args.char_emb_file)
        # Get model
        log.info('Building model...')
        #model = BiDAFCombined(word_vectors=word_vectors,
        #            char_vectors=char_vectors,
        #            hidden_size=hidden_size,
        #            use_char_emb=use_char_emb)

        model = BiDAFSelfAttendedOld(word_vectors=word_vectors,
                    char_vectors=char_vectors,
                    hidden_size=args.hidden_size,
                    drop_prob=0,
                    char_drop_prob=0,
                    use_char_emb=True) #confirm that we're actually using character embeddings
        model = nn.DataParallel(model, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()
    elif type == "charembed":
        # Get embeddings [CHARACTER EMBEDDINGS VERSION]
        log.info('Loading embeddings...')
        hidden_size = 100
        drop_prob = 0.2
        use_char_emb = True
        word_vectors = util.torch_from_json(args.word_emb_file)
        char_vectors = util.torch_from_json(args.char_emb_file)
        # Get model
        log.info('Building model...')
        #model = BiDAFChar(word_vectors=word_vectors,
        #            char_vectors=char_vectors,
        #            hidden_size=hidden_size,
        #            use_char_emb=use_char_emb)

        model = BiDAFChar(word_vectors=word_vectors,
                    char_vectors=char_vectors,
                    hidden_size=args.hidden_size,
                    drop_prob=0,
                    char_drop_prob=0,
                    use_char_emb=True) #confirm that we're actually using character embeddings
        model = nn.DataParallel(model, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()


   
    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    #place every question into a category
    category_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
        for k in gold_dict.keys():
            q = gold_dict[k]["question"]
            q = q.lower()
            category = 0
            if q.find("who") != -1:
               category = 1
            elif q.find("what") != -1:
                category = 2
            elif q.find("where") != -1:
                category = 3
            elif q.find("when") != -1:
                category = 4
            elif q.find("why") != -1:
                category = 5
            # categorize the sample

            category_dict[k] = category

    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            if type == "selfmatchattention" or type == "selfmatchcombined":
            
            # for self matching combined
                log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
            else:
                log_p1, log_p2 = model(cw_idxs, qw_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dict_categorized(gold_dict, pred_dict, args.use_squad_v2, category_dict)
        if args.use_squad_v2:
            results_list = [[('NLL', nll_meter.avg), ('F1', results[i]['f1']),('EM', results[i]['em']),('avna', results[i]["avna"]), ('total', results[i]["total"])] for i in range(5)]
        else:
            results_list = [[('NLL', nll_meter.avg), ('F1', results[i]['f1']),('EM', results[i]['em'])] for i in range(5)]

        print(results_list)




if __name__ == '__main__':
    main(get_test_args())