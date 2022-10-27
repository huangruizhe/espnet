"""Beam search module."""

from collections import defaultdict
from dataclasses import fields
from itertools import chain
import logging
from mimetypes import init
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
import torch.nn.functional as F

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.nets.scorer_interface import ScorerInterface

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis


class KeepBeamSearch(BeamSearch):
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

        self.word_list = set()
        if path_exists(wordlist_file):            
            with open(wordlist_file, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    fields = line.split()
                    if len(fields) != 2:
                        continue
                    w = fields[1]  # (kwid, kw), only consider one word query now
                    self.word_list.add(w)
        logging.info(f"There are {len(self.word_list)} words in the word list")

        self.hits = defaultdict(int)

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
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            # apply temperature and normalization
            # if self.temperature != 1.0:
            #     weighted_scores = F.log_softmax(weighted_scores / self.temperature, dim=-1)
            # add previous hyp score
            weighted_scores += hyp.score   # p( h | x) * (p1(w|h,x), p2(w|h,x), p3(w|h,x))

            for part_j in part_ids.tolist():
                part_j_token = self.token_list[part_j]
                if part_j == self.eos or part_j_token.startswith("▁"):  # the end of the last word
                    prev_word = hyp.yseq[hyp.last_word_start:]
                    prev_word = "".join([self.token_list[x] for x in prev_word])
                    prev_word = prev_word[1:]  # remove the starting symbol of word pieces
                    if prev_word in self.word_list:
                        # (hit word, step, word index)
                        # self.hits.append((prev_word, len(hyp.yseq), max(hyp.word_count - 1, 0)))
                        pos = max(hyp.word_count - 1, 0)
                        self.hits[(prev_word, pos)] += 1

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                new_scores = self.merge_scores(
                    hyp.scores, scores, j, part_scores, part_j
                )
                my_token_scores_seperate = dict()
                for k, v in new_scores.items():
                    new_word_score_k = v - hyp.scores[k]
                    my_token_scores_seperate[k] = new_word_score_k

                # Compute the normalization terms for the loglinear interpolation.
                # This is hard-wired to a specific setting. Please be careful when changing to another model
                # ac = scores['decoder'] * self.weights['decoder'] + part_scores['ctc'] * self.weights['ctc']
                # my_token_scores_seperate["ac"] = torch.exp(ac).sum().log()
                # my_token_scores_seperate["all"] = torch.exp(ac + scores['lm'][full_prev_hyp_id] * self.weights['lm']).sum().log()

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
                        last_word_start=len(hyp.yseq) if j_token.startswith("▁") else hyp.last_word_start,
                        word_count=hyp.word_count + 1 if j_token.startswith("▁") else hyp.word_count,
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
