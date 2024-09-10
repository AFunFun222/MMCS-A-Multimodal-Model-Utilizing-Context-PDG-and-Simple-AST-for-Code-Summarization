import json

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import os
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval_indicator.getScore import get_test_score, printScore
import utils
import config
import preprocess.data as data
from model_eval import Eval
from Model.model import Model
from model_test import Test as modelTest
class Train(object):

    def __init__(self, vocab_file_path=None):

        self.train_dataset = data.CodePtrDataset(code_path=config.train_code_path, nl_path=config.train_nl_path,
                                                 pdg_node_path=config.train_PDGnode_path, pdg_edge_path=config.train_PDGedge_path,
                                                 ast_node_path = config.train_astNode_path, ast_edge_path= config.train_astEdge_path)
        self.train_dataset_size = len(self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           collate_fn=lambda *args: utils.collate_fn(args,
                                                                                     code_vocab=self.code_vocab,
                                                                                     nl_vocab=self.nl_vocab,
                                                                                     pdg_vocab=self.pdg_vocab,
                                                                                     ast_vocab=self.ast_vocab), drop_last=True)

        self.code_vocab: utils.Vocab
        self.nl_vocab: utils.Vocab
        self.pdg_vocab: utils.Vocab
        self.ast_vocab: utils.Vocab
        # load vocab from given path
        if vocab_file_path:
            code_vocab_path, nl_vocab_path, pdg_vocab_path, ast_vocab_path = vocab_file_path
            self.code_vocab = utils.load_vocab_pk(code_vocab_path)
            self.nl_vocab = utils.load_vocab_pk(nl_vocab_path)
            self.pdg_vocab = utils.load_vocab_pk(pdg_vocab_path)
            self.ast_vocab = utils.load_vocab_pk(ast_vocab_path)

        # new vocab
        else:
            self.code_vocab = utils.Vocab('code_vocab')
            self.nl_vocab = utils.Vocab('nl_vocab')
            self.pdg_vocab = utils.Vocab('pdg_vocab')
            self.ast_vocab = utils.Vocab('ast_vocab')
            codes, nls, pdg_nodes, pdg_edges, ast_nodes, ast_edges = self.train_dataset.get_dataset()

            # trees = utils.load_dataset(config.train_tree_path)
            for code, nl, pn, an in zip(codes, nls, pdg_nodes, ast_nodes):
                self.code_vocab.add_sentence(code)
                self.nl_vocab.add_sentence(nl)
                self.pdg_vocab.add_sentence(pn)
                self.ast_vocab.add_sentence(an)

            self.origin_code_vocab_size = len(self.code_vocab)
            self.origin_nl_vocab_size = len(self.nl_vocab)
            self.origin_pdg_vocab_size = len(self.pdg_vocab)
            self.origin_ast_vocab_size = len(self.ast_vocab)

            # trim vocabulary
            self.code_vocab.trim(config.code_vocab_size)
            self.nl_vocab.trim(config.nl_vocab_size)
            self.pdg_vocab.trim(config.pdg_vocab_size)
            self.ast_vocab.trim(config.ast_vocab_size)

            # save vocabulary
            self.code_vocab.save(config.code_vocab_path)
            self.nl_vocab.save(config.nl_vocab_path)
            self.pdg_vocab.save(config.pdg_vocab_path)
            self.ast_vocab.save(config.ast_vocab_path)

            self.code_vocab.save_txt(config.code_vocab_txt_path)
            self.nl_vocab.save_txt(config.nl_vocab_txt_path)
            self.pdg_vocab.save_txt(config.pdg_vocab_txt_path)
            self.ast_vocab.save_txt(config.ast_vocab_txt_path)

        self.code_vocab_size = len(self.code_vocab)
        self.nl_vocab_size = len(self.nl_vocab)
        self.pdg_vocab_size = len(self.pdg_vocab)
        self.ast_vocab_size = len(self.ast_vocab)

        self.epoch = 0
        self.max_google_bleu4: float = 0
        self.max_meteor: float = 0
        self.max_rougeL: float = 0
        self.min_avg_loss: float = 100.0
        self.epoch_avg_loss: float = 100.0


        # model
        self.model = Model(code_vocab_size=self.code_vocab_size, nl_vocab_size=self.nl_vocab_size,
                           pdg_vocab_size=self.pdg_vocab_size, ast_vocab_size = self.ast_vocab_size)

        self.params = list(self.model.gat_encoder.parameters()) + list(self.model.code_encoder.parameters()) + list(self.model.decoder.parameters())

        # optimizer
        self.optimizer = Adam([
            {'params': self.model.gat_encoder.parameters(), 'lr': config.gat_encoder_lr},
            {'params': self.model.code_encoder.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.decoder.parameters(), 'lr': config.decoder_lr},
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        if config.use_lr_decay:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=config.lr_decay_every,
                                                    gamma=config.lr_decay_rate)

        checkpoint_name = 'cur.pth'
        self.load_checkpoint(checkpoint_name)

        # eval instance
        self.eval_instance = Eval()
        self.test_instance = modelTest()

        # early stopping
        self.early_stopping = None
        if config.use_early_stopping:
            self.early_stopping = utils.EarlyStopping()

        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)


    def run_train(self):
        """
        start training
        :return:
        """
        self.train_iter()

    def train_one_batch(self, batch: utils.Batch, batch_size, criterion):

        nl_batch = batch.extend_nl_batch if config.use_pointer_gen else batch.nl_batch
        self.optimizer.zero_grad()
        decoder_outputs = self.model(batch, batch_size, self.nl_vocab)
        batch_nl_vocab_size = decoder_outputs.size()[2]

        decoder_outputs = decoder_outputs.view(-1, batch_nl_vocab_size)
        nl_batch = nl_batch.view(-1)

        loss = criterion(decoder_outputs, nl_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.params, 5)

        self.optimizer.step()

        return loss

    def train_iter(self):
        plot_losses = []
        criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))
        while self.epoch < config.epoch:
            print_loss = 0
            plot_loss = 0
            last_plot_index = 0
            index = 0
            pbar = tqdm(self.train_dataloader)
            for index_batch, batch in enumerate(pbar):
                batch_size = batch.batch_size
                loss = self.train_one_batch(batch, batch_size, criterion)
                pbar.set_description("loss {}".format(loss))

                print_loss += loss.item()
                plot_loss += loss.item()
                index = index + 1

                # plot train progress details
                if index_batch % config.plot_every == 0:
                    batch_length = index_batch - last_plot_index
                    if batch_length != 0:
                        plot_loss = plot_loss / batch_length
                    plot_losses.append(plot_loss)
                    plot_loss = 0
                    last_plot_index = index_batch
            self.epoch_avg_loss = print_loss/index
            print(f"Training ===> Epoch:{self.epoch} Avg_loss:{self.epoch_avg_loss}")
            self.min_avg_loss = min(self.min_avg_loss, self.epoch_avg_loss)
            if config.use_early_stopping:
                if self.early_stopping.early_stop:
                    break

            # validate on the valid dataset every epoch
            if config.validate_during_train:
                print('\nTesting the model at the end of epoch {} on test dataset......'.format(self.epoch))
                print("The current time:{}".format(datetime.datetime.now()))
                data = self.eval_instance.run_eval(self.model)
                print(data)
                if data.google_bleu4> self.max_google_bleu4:
                    self.save_checkpoint(f"checkpoints_bleu4.pth")

            if config.use_lr_decay:
                self.lr_scheduler.step()

            self.epoch += 1
            self.save_checkpoint(f"checkpoints_cur.pth")

        plt.xlabel('every {} batches'.format(config.plot_every))
        plt.ylabel('avg loss')
        plt.plot(plot_losses)
        plt.savefig(os.path.join(config.out_dir, 'train_loss_{}.svg'.format(utils.get_timestamp())),
                    dpi=600, format='svg')
        utils.save_pickle(plot_losses, os.path.join(config.out_dir, 'plot_losses_{}.pk'.format(utils.get_timestamp())))

    def save_checkpoint(self, name):
        checkpoint = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch":self.epoch,
            "max_google_bleu4_score":self.max_google_bleu4,
            "max_meteor_score":self.max_meteor,
            "max_rougeL_score":self.max_rougeL,
            "min_avg_loss": self.min_avg_loss,
            "epoch_avg_loss": self.epoch_avg_loss
        }
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, name))
        print(f"save checkpoint:{name} success!")

    def load_checkpoint(self, name):
        checkpoint_path = os.path.join(config.checkpoint_dir, name)
        if not os.path.exists(checkpoint_path):
            print(f"{checkpoint_path} does not exist!")
            return
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epoch = checkpoint["epoch"]
        self.max_google_bleu4 = checkpoint["max_google_bleu4_score"]
        self.max_meteor_score = checkpoint["max_meteor_score"]
        self.max_rougeL_score = checkpoint["max_rougeL_score"]
        self.min_avg_loss = checkpoint["min_avg_loss"]
        print(f"load checkpoint_{name} success!")
        print("++++++++++++++++++ Parameters ++++++++++++++++++")
        print("max_google_bleu4:", self.max_google_bleu4)
        print("max_meteor_score:", self.max_meteor_score)
        print("max_rougeL_score:", self.max_rougeL_score)
        print(f"epoch:{self.epoch}")
        print("++++++++++++++++++ Parameters ++++++++++++++++++")