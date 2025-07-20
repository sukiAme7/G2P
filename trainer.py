import os
import torch
import torch.nn as nn
import numpy as np
import logging

from model import Transformer
from config import HP
from datasets import G2pdataset,collate_fn
from argparse import ArgumentParser
from torch.utils.data  import DataLoader


def evaluate(model_, devloader, crit):
    model_.eval()
    sum_loss = 0 
    with torch.no_grad():
      for batch in devloader:
          word_idxs, word_len, phoneme_idxs, phoneme_len = batch

          output, attention = model_(word_idxs.to(HP.device), phoneme_idxs[:,:-1].to(HP.device))
          out = output.view(-1, output.size(-1))
          trg = phoneme_idxs[:,1:]
          trg = trg.contiguous().view(-1)

          loss = crit(out.to(HP.device), trg.to(HP.device))
          sum_loss += loss.item()
    model_.train()
    return sum_loss/len(devloader)

def save_checkpoint(model_, epoch_, optm, checkpointpath):
    save_dict = {
      'epoch': epoch_,
      'model_state_dict': model_.state_dict(),
      'optimizer_state_dict': optm.state_dict()
    }
    torch.save(save_dict, checkpointpath)

def train():
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",  # 时间、级别、信息
        filename="logfile.log",  
        filemode="a"  #
    )
    parser = ArgumentParser(description="model training")
    parser.add_argument("--c", default=None, type=str, help="training from scratch or resume training")

    args = parser.parse_args()

    model = Transformer()
    model.to(HP.device)

    criterion =  nn.CrossEntropyLoss(ignore_index=HP.DECODER_PAD_IDX) # ignore pad idx
    opt = torch.optim.Adam(model.parameters(), lr =HP.init_lr)

    trainset = G2pdataset(HP.train_dataset_path)
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True, collate_fn=collate_fn)

    devset = G2pdataset(HP.val_dataset_path)
    dev_loader = DataLoader(devset, batch_size=4, shuffle=True, drop_last=False, collate_fn=collate_fn)

    start_epoch, step = 0,0
    if args.c:
       checkpoint = torch.load(args.c)
       model.load_state_dict(checkpoint['model_state_dict'])
       opt.load_state_dict(checkpoint['optimizer_state_dict'])
       start_epoch = checkpoint['epoch']
       print("resume from %s." % args.c)
    else:
       print("traing from scratch!")

    model.train()

    for epoch in (start_epoch, HP.epochs):
       print('Start Epoch:%d, Steps:%d' %(epoch, len(train_loader)))

       for batch in train_loader:
          word_idxs, word_len, phoneme_idxs, phoneme_len = batch
          opt.zero_grad()

          output, attention = model(word_idxs.to(HP.device), phoneme_idxs[:,:-1].to(HP.device))
          out = output.view(-1, output.size(-1))
          trg = phoneme_idxs[:,1:]
          trg = trg.contiguous().view(-1)

          loss = criterion(out.to(HP.device), trg.to(HP.device))

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), HP.gard_clip_thresh)

          opt.step()

          if not step  % HP.verbose_step:
             eval_loss = evaluate(model, dev_loader, criterion)
          if not step % HP.save_step:
             model_path = 'model_%d_%d.pth' % (epoch, step)
             save_checkpoint(model, epoch, opt, os.path.join('model_svae', model_path))
              
          step +=1
          logging.info("Epoch[%d/%d], step:%d, Train loss:%.5f, Dev loss:%.5f" %(epoch,HP.epochs, step,loss.item(),eval_loss.item()))
           

if __name__ == "__main__":
  train()




