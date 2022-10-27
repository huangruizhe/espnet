"""Beam search module."""

from collections import defaultdict
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
                    w = line.split()[0]
                    self.word_list.add(w)

    
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
        lprobs_t = torch.zeros((len(running_hyps), self.n_vocab))
        things_to_save = []
        part_ids = torch.arange(self.n_vocab, device=x.device)  # no pre-beam
        for i_hyp, hyp in enumerate(running_hyps):
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
                temp_scores = torch.full((self.n_vocab,), -float('inf'), device=x.device)
                temp_scores[part_ids] = self.weights[k] * part_scores[k]
                weighted_scores += temp_scores
            lprobs_t[i_hyp] = weighted_scores + hyp.score

            things_to_save.append(
                {
                    "scores": scores,
                    "states": states,
                }
            )
        
        inflation = 1
        topk_gs, topk_indices = torch.topk(
            lprobs_t.view(-1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                self.beam_size * inflation,
                lprobs_t.view(len(running_hyps), -1).size(1),
            ),
        )

        hyp_ids = torch.div(topk_indices, self.n_vocab, rounding_mode='floor').tolist()
        j_ids = torch.remainder(topk_indices, self.n_vocab).tolist()
        hyp_ids = hyp_ids[:self.beam_size]
        j_ids = j_ids[:self.beam_size]

        grouped_ids = defaultdict(list)
        for i_hyp, j in zip(hyp_ids, j_ids):
            grouped_ids[i_hyp].append(j)

        best_hyps = []
        # update hyps
        for i_hyp, part_ids in grouped_ids.items():
            hyp = running_hyps[i_hyp]
            scores, states = things_to_save[i_hyp]["scores"], things_to_save[i_hyp]["states"]

            part_ids = torch.tensor(part_ids)
            part_scores, part_states = self.score_partial(hyp, part_ids, x)

            for part_j, j in enumerate(part_ids):
                new_scores = self.merge_scores(
                    hyp.scores, scores, j, part_scores, part_j
                )
                my_token_scores_seperate = dict()
                for k, v in new_scores.items():
                    new_word_score_k = v - hyp.scores[k]
                    my_token_scores_seperate[k] = new_word_score_k

                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=lprobs_t[i_hyp, j],   # p(h,w | x) = p( h | x) * p(w|h,x)
                        yseq=self.append_token(hyp.yseq, j),
                        scores=new_scores,
                        states=self.merge_states(states, part_states, part_j),
                        token_scores=hyp.token_scores + [weighted_scores[j] - hyp.score],
                        token_scores_seperate=hyp.token_scores_seperate + [my_token_scores_seperate],
                    )
                )

        # sort and prune 2 x beam -> beam
        best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
            : min(len(best_hyps), self.beam_size)
        ]

        return best_hyps
    

    def search1(
        self, running_hyps: List[Hypothesis], x: torch.Tensor,
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """

        # Every hyp in running_hyps will be extended by |V| tokens.
        # This will end up with beam_size * vocab_size candidates to do topk sampling
        # This corresponed to the "step" function here: https://github.com/wouterkool/stochastic-beam-search/blob/34c43a33fd6747eb2e66a0c3cf66c0c5583a9119/fairseq/search.py#L72

        lprobs_t = torch.zeros((len(running_hyps), self.n_vocab))

        # At any step, the size of running_hyps should be self.beam_size
        # if step == 0 and len(running_hyps) == 1:
        #     lprobs_t[1:, :] = -float('inf')
        things_to_save = []

        part_ids = torch.arange(self.n_vocab, device=x.device)  # no pre-beam
        for i_hyp, hyp in enumerate(running_hyps):
            # Let's compute the score defined by the sequence model.
            # Basically, the weighted_scores can equal to any normalized distribution

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

            lprobs_t[i_hyp] = hyp.score + weighted_scores   # no normalization for the sampling distribution

            things_to_save.append(
                {
                    "scores": scores,
                    "states": states,
                }
            )

        # let's choose top-k from lprobs_t now with some heuristics
        inflation = 3   # we are allowed to have an inflation when: self.beam_size * inflation < vocab_size
        topk_gs, topk_indices = torch.topk(
            lprobs_t.view(-1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                self.beam_size * inflation,
                lprobs_t.view(len(running_hyps), -1).size(1),
            ),
        )

        hyp_ids = torch.div(topk_indices, self.n_vocab, rounding_mode='floor').tolist()
        j_ids = torch.remainder(topk_indices, self.n_vocab).tolist()

        hyp_ids = hyp_ids[:self.beam_size]
        j_ids = j_ids[:self.beam_size]

        grouped_ids = defaultdict(list)
        for i_hyp, j in zip(hyp_ids, j_ids):
            grouped_ids[i_hyp].append(j)

        best_hyps = []
        for i_hyp, part_ids in grouped_ids.items():
            # Let's compute the scores again, 
            # but this time, we actually know which id to choose.
            hyp = running_hyps[i_hyp]

            scores, states = things_to_save[i_hyp]["scores"], things_to_save[i_hyp]["states"]
            
            part_ids = torch.tensor(part_ids)
            part_scores, part_states = self.score_partial(hyp, part_ids, x)

            # update hyps
            for part_j, j in enumerate(part_ids):
                new_scores = self.merge_scores(
                    hyp.scores, scores, j, part_scores, part_j
                )
                my_token_scores_seperate = dict()
                for k, v in new_scores.items():
                    new_word_score_k = v - hyp.scores[k]
                    my_token_scores_seperate[k] = new_word_score_k

                new_hyp= Hypothesis(
                    score=lprobs_t[i_hyp, j],
                    yseq=self.append_token(hyp.yseq, j),
                    scores=new_scores,
                    states=self.merge_states(states, part_states, part_j),
                    token_scores=hyp.token_scores + [weighted_scores[j] - hyp.score],
                    token_scores_seperate=hyp.token_scores_seperate + [my_token_scores_seperate],
                )

                best_hyps.append(
                    new_hyp
                )

        best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
            : min(len(best_hyps), self.beam_size)
        ]

        return best_hyps
