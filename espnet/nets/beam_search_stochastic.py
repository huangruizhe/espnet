"""Stochastic beam search module."""

from collections import defaultdict
from itertools import chain
import logging
from re import I
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import numpy as np

import torch
import torch.nn.functional as F

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.nets.scorer_interface import ScorerInterface

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.gumbel import gumbel_like, gumbel_with_maximum


class StochasticBeamSearch(BeamSearch):
    """Stochastic beam search implementation."""

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
        temperature: float = 1.0,
        beam_search_mode: str = "topk",
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

        self.temperature = temperature

        # three modes of beam search
        self.stochastic = False
        self.topk = False
        self.sampling = False
        if beam_search_mode == "stochastic":
            self.stochastic = True
        elif beam_search_mode == "sampling":
            self.sampling = True
        elif beam_search_mode == "topk":
            self.topk = True
        else:
            logging.error(f"Invalid beam_search_mode={beam_search_mode}")
            raise NotImplementedError

    def search(
        self, running_hyps: List[Hypothesis], x: torch.Tensor, step: int
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

        # These are the lob probs we will use to do top-k sampling:
        #   - lprobs: lob probability for each candidate
        # Note that: 
        #   - each of the probabilities should sum-to-one, to ensure a valid sampling process
        #   - we sample without replacement by the total prob mass
        lprobs = torch.zeros((len(running_hyps), self.n_vocab))  # device=x.device
        lprobs_t = torch.zeros((len(running_hyps), self.n_vocab))

        uniform_tmp = torch.log(torch.full((self.n_vocab,), 1/self.n_vocab))
        biased_tmp = torch.full((self.n_vocab,), -1e6)  # -float('inf')
        biased_tmp[self.eos] = 0

        # At any step, the size of running_hyps should be self.beam_size
        # if step == 0 and len(running_hyps) == 1:
        #     lprobs_t[1:, :] = -float('inf')
        things_to_save = []

        part_ids = torch.arange(self.n_vocab, device=x.device)  # no pre-beam
        hyp_G = lprobs_t.new(size=(len(running_hyps),))
        for i_hyp, hyp in enumerate(running_hyps):
            # Let's compute the score defined by the sequence model.
            # Basically, the weighted_scores can equal to any normalized distribution

            hyp_G[i_hyp] = hyp.G
            if step > 0 and hyp.yseq[-1] == self.eos:
                lprobs_t[i_hyp] = hyp.score_t + biased_tmp
                things_to_save.append(None)
                continue

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
                val = part_scores[k].min().item() - 2.0  # TODO: This 2.0 is a tunable number, which provides an estimates for un-evaluated ctc scores
                part_scores_full = torch.full((self.n_vocab,), val, device=part_scores[k].device)
                part_scores_full[part_ids] = part_scores[k]
                weighted_scores += self.weights[k] * part_scores_full

            # This is cumulative prob for each candidate
            lprobs_t[i_hyp] = hyp.score_t + F.log_softmax(weighted_scores / self.temperature, dim=-1)
            weighted_scores = hyp.score + weighted_scores

            things_to_save.append(
                {
                    "scores": scores,
                    "states": states,
                    "weighted_scores": weighted_scores
                }
            )

        # let's do the top-k sampling now
        if self.stochastic:
            if step == 0:    
                cand_scores = gumbel_like(lprobs_t) + lprobs_t
            else:
                cand_scores, _ = gumbel_with_maximum(lprobs_t, hyp_G, -1)
        else:
            cand_scores = lprobs_t
        
        if self.stochastic or self.topk:
            topk_gs, topk_indices = torch.topk(
                cand_scores.view(-1),
                k=min(
                    # Take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.
                    self.beam_size,  #  * 2 # TODO
                    cand_scores.view(len(running_hyps), -1).size(1),
                ),
            )
        elif self.sampling:
            raise NotImplementedError
        else:
            raise NotImplementedError

        hyp_ids = torch.div(topk_indices, self.n_vocab, rounding_mode='floor').tolist()
        j_ids = torch.remainder(topk_indices, self.n_vocab).tolist()
        grouped_ids = defaultdict(list)
        for i_hyp, j in zip(hyp_ids, j_ids):
            grouped_ids[i_hyp].append(j)

        best_hyps = []
        for i_hyp, part_ids in grouped_ids.items():
            # Let's compute the scores again, 
            # but this time, we actually know which id to choose.
            hyp = running_hyps[i_hyp]

            if step > 0 and hyp.yseq[-1] == self.eos:
                best_hyps.append(hyp)
                continue

            scores, states = things_to_save[i_hyp]["scores"], things_to_save[i_hyp]["states"]
            weighted_scores = things_to_save[i_hyp]["weighted_scores"]
            
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
                    score=weighted_scores[j],
                    yseq=self.append_token(hyp.yseq, j),
                    scores=new_scores,
                    states=self.merge_states(states, part_states, part_j),
                    token_scores=hyp.token_scores + [weighted_scores[j] - hyp.score],
                    token_scores_seperate=hyp.token_scores_seperate + [my_token_scores_seperate],
                    G=cand_scores[i_hyp][j],
                    score_t=lprobs_t[i_hyp][j],
                )

                best_hyps.append(
                    new_hyp
                )
                # if np.isnan(best_hyps[-1].states["ctc"][1].sum()):
                #     logging.error("here")

        # sort and prune 2 x beam -> beam
        best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
            : min(len(best_hyps), self.beam_size)
        ]

        return best_hyps

    def forward(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
    ) -> List[Hypothesis]:
        """Perform stochastic beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        ended_hyps = []
        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, x, i)

            # post process of one iteration
            # running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)

            # Option1: Strictly sample beam_size from leaves:
            # ended_hyps = []
            # running_hyps = self.post_process_for_sbs(i, maxlen, maxlenratio, best, ended_hyps)

            # Option2: Keep beam to be beam_size, but we may have more leaves
            running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)            

            # end detection
            logging.debug(f"the number of ended hypotheses: {len(ended_hyps)}")
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logging.info(f"end detected at {i}")
                break

            if len(running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(running_hyps)}")

        # ended_hyps = []
        # _ = self.post_process_for_sbs(i, maxlen, maxlenratio, running_hyps, ended_hyps)

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def post_process_for_sbs(
        self,
        i: int,
        maxlen: int,
        maxlenratio: float,
        running_hyps: List[Hypothesis],
        ended_hyps: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            List[Hypothesis]: The new running hypotheses.

        """
        logging.debug(f"the number of running hypotheses: {len(running_hyps)}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]])
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            running_hyps_ = []
            for hyp in running_hyps:
                if hyp.yseq[-1] != self.eos:
                    running_hyps_.append(hyp._replace(yseq=self.append_token(hyp.yseq, self.eos)))
                else:
                    running_hyps_.append(hyp)
            running_hyps = running_hyps_
        
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                hyp = hyp._replace(is_done=True)
                ended_hyps.append(hyp)
        return running_hyps

    def post_process_for_sbs2(
        self,
        i: int,
        maxlen: int,
        maxlenratio: float,
        running_hyps: List[Hypothesis],
        ended_hyps: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            List[Hypothesis]: The new running hypotheses.

        """
        logging.debug(f"the number of running hypotheses: {len(running_hyps)}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]])
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            running_hyps = [
                h._replace(yseq=self.append_token(h.yseq, self.eos))
                for h in running_hyps
            ]
        
        for hyp in running_hyps:
            end_pos = hyp.yseq.tolist().index(self.eos)
            hyp._replace(yseq=hyp.yseq[: end_pos + 1])

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        return remained_hyps

    def forward_original(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
    ) -> List[Hypothesis]:
        """Perform stochastic beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        ended_hyps = []
        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, x, i)
            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)
            # end detection
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logging.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(running_hyps)}")

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps