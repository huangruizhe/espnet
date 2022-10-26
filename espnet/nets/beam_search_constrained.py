"""Beam search module."""

from itertools import chain
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union
from xmlrpc.client import Boolean
import numpy as np
from os.path import exists as path_exists

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.nets.scorer_interface import ScorerInterface

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis


class Trie:
    def __init__ (
        self,
        wordlist,
        _end="$"
    ):
        self._end = _end
        self.root = self.make_letter_trie(wordlist)

    def make_letter_trie(self, wordlist):
        root = dict()
        dict_count = 1
        for word in wordlist:
            current_dict = root
            for letter in word:   # word is a string
                current_dict = current_dict.setdefault(letter, {})
                if len(current_dict) == 0:
                    dict_count += 1
            current_dict[self._end] = self._end  # TODO: put something useful here for the scorer
        logging.info(f"Done making trie with {dict_count} nodes for {len(wordlist)} dictionary words")
        return root
    
    def in_trie_word(self, word):
        current_dict = self.root
        for letter in word:
            if letter not in current_dict:
                return False
            current_dict = current_dict[letter]
        return self._end in current_dict  # must be a full word
    
    def in_trie_prefix(self, word):
        current_dict = self.root
        for letter in word:
            if letter not in current_dict:
                return False
            current_dict = current_dict[letter]
        return True
    
    def __contains__(self, prefix):
        return self.in_trie_prefix(prefix)


class ConstrainedBeamSearch(BeamSearch):
    """Constrained beam search implementation."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        wordlist_file: str = None,
    ):
        """Initialize beam search.

        Args:
            scorers (dict[str, ScorerInterface]): Dict of decoder modules
                e.g., Decoder, CTCPrefixScorer, LM
                The scorer will be ignored if it is `None`
            weights (dict[str, float]): Dict of weights for each scorers
                The scorer will be ignored if its weight is 0
            beam_size (int): The number of hypotheses kept during search
            vocab_size (int): The number of vocabulary
            sos (int): Start of sequence id
            eos (int): End of sequence id
            token_list (list[str]): List of tokens for debug log
            pre_beam_score_key (str): key of scores to perform pre-beam search
            pre_beam_ratio (float): beam size in the pre-beam search
                will be `int(pre_beam_ratio * beam_size)`

        """
        super().__init__(
            scorers=scorers,
            weights=weights,
            beam_size=beam_size,
            vocab_size=vocab_size,
            sos=sos,
            eos=eos,
            token_list=token_list,
            pre_beam_ratio=pre_beam_ratio,
            pre_beam_score_key=pre_beam_score_key,
        )

        self.word_list = set([""])
        if path_exists(wordlist_file):            
            with open(wordlist_file, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    w = line.split()[0]
                    self.word_list.add(w)
        self.trie = Trie(self.word_list)

    def check_extension(self, hyp, ids_to_append, get_id_func=lambda x: x):
        # up to this point, every hyp is ended with a legal word prefix

        constraints_satisfied = []

        prefix = hyp.yseq[hyp.last_word_start:]
        if len(prefix) == 1 and prefix[0] == self.eos:
            prefix = ""
        else:
            prefix = "".join([self.token_list[x] for x in prefix])
            prefix = prefix[1:]  # remove the starting symbol of word pieces

        for id in ids_to_append:
            token_id = get_id_func(id)
            if token_id == self.eos:
                constraints_satisfied.append(True)
                continue

            new_token = self.token_list[token_id]
            if new_token.startswith("▁"):
                if not self.trie.in_trie_word(prefix):  # check the finishing word -- which may be an oov but also a prefix of another word
                    constraints_satisfied.append(False)
                    continue  # no need to consier this hypothesis anymore
                new_prefix = new_token[1:]
            else:
                new_prefix = prefix + new_token
            
            if new_prefix in self.trie:
                constraints_satisfied.append(True)
            else:
                constraints_satisfied.append(False)
        assert len(constraints_satisfied) == len(ids_to_append)
        return constraints_satisfied

    def search(
        self, running_hyps: List[Hypothesis], x: torch.Tensor
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        best_hyps = []
        part_ids = torch.arange(self.n_vocab, device=x.device)  # no pre-beam
        for hyp in running_hyps:    # accumulated in this 
            hyp_pre_beam_size = self.pre_beam_size
            hyp_beam_size = self.beam_size
            trials = 0
            trials_max = 0
            hyp_pre_beam_size_max = 256
            while True:
                # scoring
                weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)
                scores, states = self.score_full(hyp, x)
                for k in self.full_scorers:
                    weighted_scores += self.weights[k] * scores[k]
                # partial scoring
                if self.do_pre_beam:
                    pre_beam_scores = (
                        weighted_scores
                        if self.pre_beam_score_key == "full"
                        else scores[self.pre_beam_score_key]
                    )

                    pre_trials = 0
                    pre_trials_max = 1
                    if hyp_pre_beam_size > hyp_pre_beam_size_max:
                        hyp_pre_beam_size = hyp_pre_beam_size_max
                    while True:
                        part_ids = torch.topk(pre_beam_scores, hyp_pre_beam_size)[1]

                        constraints_satisfied = self.check_extension(hyp, part_ids.tolist())
                        if sum(constraints_satisfied) >= self.pre_beam_size \
                            or pre_trials >= pre_trials_max \
                            or hyp_pre_beam_size == hyp_pre_beam_size_max:
                            part_ids = [id for id, sat in zip(part_ids.tolist(), constraints_satisfied) if sat is True]
                            part_ids = part_ids[:self.pre_beam_size]
                            part_ids = torch.tensor(part_ids)
                            break
                        else:
                            # hyp_pre_beam_size *= 2
                            hyp_pre_beam_size = hyp_pre_beam_size_max
                            pre_trials += 1

                if part_ids.shape[0] == 0:
                    part_ids2 = []
                    break

                part_scores, part_states = self.score_partial(hyp, part_ids, x)
                for k in self.part_scorers:
                    weighted_scores[part_ids] += self.weights[k] * part_scores[k]
                # add previous hyp score
                weighted_scores += hyp.score

                part_ids2 = []
                for j, part_j in zip(*self.beam(weighted_scores, part_ids, beam_size=hyp_beam_size)):
                    # j is the new token to append
                    part_ids2.append((j, part_j))
                constraints_satisfied = self.check_extension(hyp, part_ids2, get_id_func=lambda x: x[0])

                if sum(constraints_satisfied) >= self.beam_size or trials >= trials_max:
                    part_ids2 = [id for id, sat in zip(part_ids2, constraints_satisfied) if sat is True]
                    part_ids2 = part_ids2[:self.beam_size]
                    break
                elif hyp_beam_size < hyp_pre_beam_size:
                    hyp_beam_size *= 2
                    trials += 1
                else:
                    hyp_pre_beam_size *= 2
                    trials += 1

            if len(part_ids2) == 0:
                logging.debug(f"hyp will be discarded: {''.join([self.token_list[x] for x in hyp.yseq[1:]])}")

            # update hyps
            for j, part_j in part_ids2:
                new_scores = self.merge_scores(
                    hyp.scores, scores, j, part_scores, part_j
                )
                my_token_scores_seperate = dict()
                for k, v in new_scores.items():
                    new_word_score_k = v - hyp.scores[k]
                    my_token_scores_seperate[k] = new_word_score_k
                
                j_token = self.token_list[j]

                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],   # p(h,w | x) = p( h | x) * p(w|h,x)
                        yseq=self.append_token(hyp.yseq, j),
                        scores=new_scores,
                        states=self.merge_states(states, part_states, part_j),
                        token_scores=hyp.token_scores + [weighted_scores[j] - hyp.score],
                        token_scores_seperate=hyp.token_scores_seperate + [my_token_scores_seperate],
                        last_word_start=len(hyp.yseq) if j_token.startswith("▁") else hyp.last_word_start
                    )
                )
                # if np.isnan(best_hyps[-1].states["ctc"][1].sum()):
                #     logging.error("here")

            # sort and prune 2 x beam -> beam
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]

            # sorted_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)
            # temperature = 
            # best_hyps = 
        return best_hyps

