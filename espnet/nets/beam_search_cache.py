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


class CacheEntry():
    def __init__(self):
        self.count = 0
        self.lifetime = 0
    
    def check(self):
        self.count += 1
        return self.count


class Cache():
    def __init__(self, life_thres=4):
        self.life_thres = life_thres
        self.entries = dict()   # word_prefix => cache entry

    def check(self, word_prefix):
        entry = self.enties.setdefault(word_prefix, CacheEntry())
        return entry.check()
    
    def forget(self):
        to_forget = list()
        n_entries_before = len(self.entries)
        for word_prefix, entry in self.entries.items():
            entry.lifetime += 1
            if entry.lifetime >= life_thres:
                to_forget.append(word_prefix)
        for forget_prefix in to_forget:
            self.entries.pop(forget_prefix)
        logging.debug(f"There were {n_entries_before} entries, {len(to_forget)} are forgotten now, {len(self.entries)} are left.")


class CachedBeamSearch(BeamSearch):
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
        life_thres: int = 4,
        hits_thres: int = 4,
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

        self.hits_thres = hits_thres
        self.cache = Cache(life_thres=life_thres)

    def select_topk(self, running_hyps, hyp_ids, j_ids):
        topk_hyp_ids = []
        topk_j_ids = []
        for i_hyp, j in zip(hyp_ids, j_ids):
            hyp = running_hyps[i_hyp]

            j_token = self.token_list[j]
            if j_token.startswith("▁"):
                new_word_prefix = j_token[1:]
            else:
                new_word_prefix = hyp.last_word + j_token
            
            rs = self.cache.check(new_word_prefix)
            if rs <= self.hits_thres:
                topk_hyp_ids.append(i_hyp)
                topk_j_ids.append(j)
            
                if len(topk_hyp_ids) >= self.beam_size:
                    break
        return topk_hyp_ids, topk_j_ids
    
    def last_resort(self, topk_hyp_ids, topk_j_ids, running_hyps, hyp_ids, j_ids):
        seen = set(zip(topk_hyp_ids, topk_j_ids))
        i = 0
        while len(topk_hyp_ids) < self.beam_size:
            if (hyp_ids[i], j_ids[i]) not in seen:
                topk_hyp_ids.append(hyp_ids[i])
                topk_j_ids.append(j_ids[i])
            i += 1

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
        
        inflation = 2
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
        
        # select your top-k among the many candidates
        # hyp_ids = hyp_ids[:self.beam_size]
        # j_ids = j_ids[:self.beam_size]
        topk_hyp_ids, topk_j_ids = self.select_topk(running_hyps, hyp_ids, j_ids)
        if len(topk_hyp_ids) < self.beam_size:
            self.last_resort(topk_hyp_ids, topk_j_ids, running_hyps, hyp_ids, j_ids)

        grouped_ids = defaultdict(list)
        for i_hyp, j in zip(topk_hyp_ids, topk_j_ids):
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

                j_token = self.token_list[j]
                if j_token.startswith("▁"):
                    last_word = j_token
                else:
                    last_word = hyp.last_word + j

                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=lprobs_t[i_hyp, j],   # p(h,w | x) = p( h | x) * p(w|h,x)
                        yseq=self.append_token(hyp.yseq, j),
                        scores=new_scores,
                        states=self.merge_states(states, part_states, part_j),
                        token_scores=hyp.token_scores + [weighted_scores[j] - hyp.score],
                        token_scores_seperate=hyp.token_scores_seperate + [my_token_scores_seperate],
                        last_word_start=len(hyp.yseq) if j_token.startswith("▁") else hyp.last_word_start,
                        last_word=last_word,
                    )
                )

        # sort and prune 2 x beam -> beam
        best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
            : min(len(best_hyps), self.beam_size)
        ]
        self.cache.forget()

        return best_hyps
    

    