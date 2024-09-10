import json
import torch
from torch.utils.data import DataLoader
import preprocess.data as data
import utils
import config
# from eval_indicator.getScore import get_eval_score
from eval_indicator.getScore import get_test_score, printScore
from BeamNode import BeamNode
class Eval(object):

    def __init__(self):

        # vocabulary
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)
        self.pdg_vocab= utils.load_vocab_pk(config.pdg_vocab_path)
        self.pdg_vocab_size = len(self.pdg_vocab)
        self.ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
        self.ast_vocab_size = len(self.ast_vocab)
        # dataset
        self.dataset = data.CodePtrDataset(code_path=config.valid_code_path, nl_path=config.valid_nl_path,
                                           pdg_node_path=config.valid_PDGnode_path, pdg_edge_path=config.valid_PDGedge_path,
                                                 ast_node_path = config.valid_astNode_path, ast_edge_path= config.valid_astEdge_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.eval_batch_size,
                                     collate_fn=lambda *args: utils.collate_fn(args, code_vocab=self.code_vocab,
                                                                               pdg_vocab=self.pdg_vocab,
                                                                               nl_vocab=self.nl_vocab,
                                                                               ast_vocab = self.ast_vocab,
                                                                               raw_nl=True),
                                     drop_last=True)  # 直接丢掉？？？？？？？？？？？？？？？

        self.model = None

    def run_eval(self, model):
        # bleu = self.eval_iter()
        # return bleu
        self.model = model
        self.model.model_openEval()
        data = self.eval_iter()
        self.model.model_openTrain()
        return data

    def eval_one_batch(self, batch: utils.Batch, batch_size):
        """logger
        evaluate one batch
        :param batch:
        :param batch_size:
        :param criterion:
        :return:
        """
        with torch.no_grad():
            nl_batch = batch.nl_batch
            code_outputs, gat_outputs, ast_outputs, decoder_hidden = self.model(batch, batch_size, self.nl_vocab,is_test=True)  # [T, B, nl_vocab_size]
            # [300,32,512] [20,32,512] [1,32,512]
            extend_type_batch = None
            extra_zeros = None
            if config.use_pointer_gen:
                extend_type_batch, _, extra_zeros = batch.get_pointer_gen_input()

            batch_sentences = self.greedy_decode(batch_size=batch_size,
                                                 code_outputs=code_outputs,
                                                 gat_outputs=gat_outputs,
                                                 ast_outputs=ast_outputs,
                                                 decoder_hidden=decoder_hidden,
                                                 extend_type_batch=extend_type_batch,
                                                 extra_zeros=extra_zeros)  # 问题出现在这里？ 为什么只有只是1呀？
            # print("batch_oovs:", batch.batch_oovs)
            # print("batch_sentences:", batch_sentences)
            candidates = self.translate_indices(batch_sentences, batch.batch_oovs)

            # c_bleu, s_bleu, bleu4 = get_eval_score(references=nl_batch, candidates=candidates)

            return nl_batch, candidates

    def eval_iter(self):
        """
        evaluate model on self.dataset
        :return: scores
        """
        # self.model.eval()
        refereces = []
        candidates = []
        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch.batch_size
            refs, cands = self.eval_one_batch(batch,batch_size)
            refereces.extend(refs)
            candidates.extend(cands)
        # c_bleu, s_bleu, bleu1_score, bleu2_score, bleu3_score, bleu4_score = get_eval_score(references=refereces,candidates=candidates)
        data = get_test_score(references=refereces,candidates=candidates)

        with open(config.valid_save, "w", encoding="utf-8") as writer:
            for index in range(len(refereces)):
                ref = ' '.join(refereces[index])
                cad = ' '.join(candidates[index])
                mp = {
                    "reference": ref,
                    "candidate": cad
                }
                json.dump(mp, writer)
                writer.write('\n')
        # self.model.train()

        return data

    def greedy_decode(self, batch_size, code_outputs: torch.Tensor, gat_outputs: torch.Tensor,ast_outputs: torch.Tensor,
                      decoder_hidden: torch.Tensor, extend_type_batch, extra_zeros):
        """
        beam decode for one batch, feed one batch for decoder
        :param batch_size:
        :param source_outputs: [T, B, H]
        :param code_outputs: [T, B, H]
        :param ast_outputs: [T, B, H]
        :param decoder_hidden: [1, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: batch_sentences, [B, config.beam_top_sentence]
        """
        batch_sentences = []

        for index_batch in range(batch_size):
            # for each input sentence
            single_decoder_hidden = decoder_hidden[:, index_batch, :].unsqueeze(1)  # [1, 1, H]
            single_code_output = code_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]
            single_gat_output = gat_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]
            single_ast_output = ast_outputs[:, index_batch, :].unsqueeze(1)

            # single_tree_output = tree_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]

            single_extend_type = None
            single_extra_zeros = None

            if config.use_pointer_gen:
                single_coverage = torch.zeros((1, config.max_code_length), device=config.device)  # [1, T]
                single_extend_type = extend_type_batch[index_batch]
                if extra_zeros is not None:
                    single_extra_zeros = extra_zeros[index_batch]

            root = BeamNode(sentence_indices=[utils.get_sos_index(self.nl_vocab)],
                            log_probs=[0.0],
                            hidden=single_decoder_hidden)

            current_nodes = [root]  # list of nodes to be further extended
            final_nodes = []  # list of end nodes

            for step in range(config.max_decode_steps):
                if len(current_nodes) == 0:
                    break

                candidate_nodes = []  # list of nodes to be extended next step

                feed_inputs = []
                feed_hidden = []

                # B = len(current_nodes) except eos
                extend_nodes = []
                for node in current_nodes:
                    # if current node is EOS
                    if node.word_index() == utils.get_eos_index(self.nl_vocab):
                        final_nodes.append(node)
                        # if number of final nodes reach the beam width
                        if len(final_nodes) >= 1:
                            break
                        continue

                    extend_nodes.append(node)

                    decoder_input = utils.tune_up_decoder_input(node.word_index(), self.nl_vocab)

                    single_decoder_hidden = node.hidden.clone().detach()  # [1, 1, H]

                    feed_inputs.append(decoder_input)  # [B]
                    feed_hidden.append(single_decoder_hidden)  # B x [1, 1, H]

                if len(extend_nodes) == 0:
                    break

                feed_batch_size = len(feed_inputs)
                feed_code_outputs = single_code_output.repeat(1, feed_batch_size, 1)
                feed_gat_outputs = single_gat_output.repeat(1, feed_batch_size, 1)
                feed_ast_outputs = single_ast_output.repeat(1, feed_batch_size, 1)

                feed_extend_type = None
                feed_extra_zeros = None

                if config.use_pointer_gen:
                    feed_extend_type = single_extend_type.repeat(feed_batch_size, 1)
                    if single_extra_zeros is not None:
                        feed_extra_zeros = single_extra_zeros.repeat(feed_batch_size, 1)

                feed_inputs = torch.tensor(feed_inputs, device=config.device)  # [B] [1]
                feed_hidden = torch.stack(feed_hidden, dim=2).squeeze(0)  # [1, B, H] [1,1,512]

                # decoder_outputs: [B, nl_vocab_size]
                # new_decoder_hidden: [1, B, H]
                # attn_weights: [B, 1, T]
                # coverage: [B, T]
                decoder_outputs, new_decoder_hidden = self.model.decoder(inputs=feed_inputs,
                                                                         last_hidden=feed_hidden,
                                                                         code_outputs=feed_code_outputs,
                                                                         gat_outputs=feed_gat_outputs,
                                                                         ast_outputs=feed_ast_outputs,
                                                                         extend_type_batch=feed_extend_type,
                                                                         extra_zeros=feed_extra_zeros)

                # get top k words
                # log_probs: [B, beam_width]
                # word_indices: [B, beam_width]
                batch_log_probs, batch_word_indices = decoder_outputs.topk(1)

                for index_node, node in enumerate(extend_nodes):
                    log_probs = batch_log_probs[index_node]
                    word_indices = batch_word_indices[index_node]
                    hidden = new_decoder_hidden[:, index_node, :].unsqueeze(1)

                    for i in range(1):
                        log_prob = log_probs[i]
                        word_index = word_indices[i].item()

                        new_node = node.extend_node(word_index=word_index,
                                                    log_prob=log_prob,
                                                    hidden=hidden)
                        candidate_nodes.append(new_node)

                # sort candidate nodes by log_prb and select beam_width nodes
                candidate_nodes = sorted(candidate_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
                current_nodes = candidate_nodes[: 1]

            final_nodes += current_nodes
            final_nodes = sorted(final_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
            final_nodes = final_nodes[: config.beam_top_sentences]

            sentences = []
            for final_node in final_nodes:
                sentences.append(final_node.sentence_indices)

            batch_sentences.append(sentences)

        return batch_sentences

    def translate_indices(self, batch_sentences, batch_oovs: list):
        """
        translate indices to words for one batch
        :param batch_sentences: [B, config.beam_top_sentences, sentence_length]
        :param batch_oovs: list of oov words list for one batch, None if not use pointer gen, [B, oov_num(variable)]
        :return:
        """
        # print("nl_vocab.index2word:", self.nl_vocab.index2word)
        batch_words = []
        for index_batch, sentences in enumerate(batch_sentences):
            words = []

            for indices in sentences:  # index 的复数
                for index in indices:  # indices is a list of length 1, only loops once
                    if index not in self.nl_vocab.index2word:
                        assert batch_oovs is not None
                        oovs = batch_oovs[index_batch]
                        oov_index = index - self.nl_vocab_size
                        try:
                            word = oovs[oov_index]
                        except IndexError:
                            word = '<UNK>'
                    else:
                        word = self.nl_vocab.index2word[index]
                    if utils.is_unk(word) or not utils.is_special_symbol(word):
                        words.append(word)
            batch_words.append(words)
        return batch_words


