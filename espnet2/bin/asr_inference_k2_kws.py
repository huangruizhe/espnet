#!/usr/bin/env python3
import argparse
import datetime
import logging
from pathlib import Path
import sys
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import List


import k2
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List
import yaml

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fst.lm_rescore import nbest_am_lm_scores
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def indices_to_split_size(indices, total_elements: int = None):
    """convert indices to split_size

    During decoding, the api torch.tensor_split should be used.
    However, torch.tensor_split is only available with pytorch >= 1.8.0.
    So torch.split is used to pass ci with pytorch < 1.8.0.
    This fuction is used to prepare input for torch.split.
    """
    if indices[0] != 0:
        indices = [0] + indices

    split_size = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
    if total_elements is not None and sum(split_size) != total_elements:
        split_size.append(total_elements - sum(split_size))
    return split_size


# copied from:
# https://github.com/k2-fsa/snowfall/blob/master/snowfall/training/ctc_graph.py#L13
def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    """Build CTC topology.

    A token which appears once on the right side (i.e. olabels) may
    appear multiple times on the left side (ilabels), possibly with
    epsilons in between.
    When 0 appears on the left side, it represents the blank symbol;
    when it appears on the right side, it indicates an epsilon. That
    is, 0 has two meanings here.
    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = ""
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                arcs += f"{i} {i} {tokens[i]} 0 0.0\n"
            else:
                arcs += f"{i} {j} {tokens[j]} {tokens[j]} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


# Modified from: https://github.com/k2-fsa/snowfall/blob/master/snowfall/common.py#L309
def get_texts(best_paths: k2.Fsa) -> List[List[int]]:
    """Extract the texts from the best-path FSAs.

     Args:
         best_paths:  a k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
                  containing multiple FSAs, which is expected to be the result
                  of k2.shortest_path (otherwise the returned values won't
                  be meaningful).  Must have the 'aux_labels' attribute, as
                a ragged tensor.
    Return:
        Returns a list of lists of int, containing the label sequences we
        decoded.
    """
    # remove any 0's or -1's (there should be no 0's left but may be -1's.)

    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape())

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values())
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    return aux_labels.tolist()


def _intersect_device(
    a_fsas: k2.Fsa,
    b_fsas: k2.Fsa,
    b_to_a_map: torch.Tensor,
    sorted_match_a: bool,
    batch_size: int = 50,
) -> k2.Fsa:
    """This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.
    The arguments and return value of this function are the same as
    :func:`k2.intersect_device`.
    """
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(
            a_fsas, b_fsas, b_to_a_map=b_to_a_map, sorted_match_a=sorted_match_a
        )

    num_batches = (num_fsas + batch_size - 1) // batch_size
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index_fsa(b_fsas, indexes)
        b_to_a = k2.index_select(b_to_a_map, indexes)
        path_lattice = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lattice)

    return k2.cat(ans)


def get_lattice(
    nnet_output: torch.Tensor,
    decoding_graph: k2.Fsa,
    supervision_segments: torch.Tensor,
    search_beam: float,
    output_beam: float,
    min_active_states: int,
    max_active_states: int,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """Get the decoding lattice from a decoding graph and neural
    network output.
    Args:
      nnet_output:
        It is the output of a neural model of shape `(N, T, C)`.
      decoding_graph:
        An Fsa, the decoding graph. It can be either an HLG
        (see `compile_HLG.py`) or an H (see `k2.ctc_topo`).
      supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice


class Nbest(object):
    """
    An Nbest object contains two fields:
        (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
                 Its axes are [path][state][arc]
        (2) shape. Its type is :class:`k2.RaggedShape`.
                   Its axes are [utt][path]
    The field `shape` has two axes [utt][path]. `shape.dim0` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.
    Caution:
      Don't be confused by the name `Nbest`. The best in the name `Nbest`
      has nothing to do with `best scores`. The important part is
      `N` in `Nbest`, not `best`.
    """

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        """
        Args:
          fsa:
            An FsaVec with axes [path][state][arc]. It is expected to contain
            a list of **linear** FSAs.
          shape:
            A ragged shape with two axes [utt][path].
        """
        assert len(fsa.shape) == 3, f"fsa.shape: {fsa.shape}"
        assert shape.num_axes == 2, f"num_axes: {shape.num_axes}"

        if fsa.shape[0] != shape.tot_size(1):
            raise ValueError(
                f"{fsa.shape[0]} vs {shape.tot_size(1)}\n"
                "Number of FSAs in `fsa` does not match the given shape"
            )

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = "Nbest("
        s += f"Number of utterances:{self.shape.dim0}, "
        s += f"Number of Paths:{self.fsa.shape[0]})"
        return s

    @staticmethod
    def from_lattice(
        lattice: k2.Fsa,
        num_paths: int,
        use_double_scores: bool = True,
        nbest_scale: float = 0.5,
    ) -> "Nbest":
        """Construct an Nbest object by **sampling** `num_paths` from a lattice.
        Each sampled path is a linear FSA.
        We assume `lattice.labels` contains token IDs and `lattice.aux_labels`
        contains word IDs.
        Args:
          lattice:
            An FsaVec with axes [utt][state][arc].
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
          use_double_scores:
            True to use double precision in :func:`k2.random_paths`.
            False to use single precision.
          scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
        Returns:
          Return an Nbest instance.
        """
        saved_scores = lattice.scores.clone()
        lattice.scores *= nbest_scale
        # path is a ragged tensor with dtype torch.int32.
        # It has three axes [utt][path][arc_pos]
        path = k2.random_paths(
            lattice, num_paths=num_paths, use_double_scores=use_double_scores
        )
        lattice.scores = saved_scores

        # word_seq is a k2.RaggedTensor sharing the same shape as `path`
        # but it contains word IDs. Note that it also contains 0s and -1s.
        # The last entry in each sublist is -1.
        # It axes is [utt][path][word_id]
        if isinstance(lattice.aux_labels, torch.Tensor):
            word_seq = k2.ragged.index(lattice.aux_labels, path)
        else:
            word_seq = lattice.aux_labels.index(path)
            word_seq = word_seq.remove_axis(word_seq.num_axes - 2)
        word_seq = word_seq.remove_values_leq(0)

        # Each utterance has `num_paths` paths but some of them transduces
        # to the same word sequence, so we need to remove repeated word
        # sequences within an utterance. After removing repeats, each utterance
        # contains different number of paths
        #
        # `new2old` is a 1-D torch.Tensor mapping from the output path index
        # to the input path index.
        _, _, new2old = word_seq.unique(
            need_num_repeats=False, need_new2old_indexes=True
        )

        # kept_path is a ragged tensor with dtype torch.int32.
        # It has axes [utt][path][arc_pos]
        kept_path, _ = path.index(new2old, axis=1, need_value_indexes=False)

        # utt_to_path_shape has axes [utt][path]
        utt_to_path_shape = kept_path.shape.get_layer(0)

        # Remove the utterance axis.
        # Now kept_path has only two axes [path][arc_pos]
        kept_path = kept_path.remove_axis(0)

        # labels is a ragged tensor with 2 axes [path][token_id]
        # Note that it contains -1s.
        labels = k2.ragged.index(lattice.labels.contiguous(), kept_path)

        # Remove -1 from labels as we will use it to construct a linear FSA
        labels = labels.remove_values_eq(-1)

        if isinstance(lattice.aux_labels, k2.RaggedTensor):
            # lattice.aux_labels is a ragged tensor with dtype torch.int32.
            # It has 2 axes [arc][word], so aux_labels is also a ragged tensor
            # with 2 axes [arc][word]
            aux_labels, _ = lattice.aux_labels.index(
                indexes=kept_path.values, axis=0, need_value_indexes=False
            )
        else:
            assert isinstance(lattice.aux_labels, torch.Tensor)
            aux_labels = k2.index_select(lattice.aux_labels, kept_path.values)
            # aux_labels is a 1-D torch.Tensor. It also contains -1 and 0.

        fsa = k2.linear_fsa(labels)
        fsa.aux_labels = aux_labels
        # Caution: fsa.scores are all 0s.
        # `fsa` has only one extra attribute: aux_labels.
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice, get 1-best
        path from the resulting FsaVec, and return a new Nbest object.
        The purpose of this function is to attach scores to an Nbest.
        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]. If it has `aux_labels`, then
            we assume its `labels` are token IDs and `aux_labels` are word IDs.
            If it has only `labels`, we assume its `labels` are word IDs.
          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`. Also, its `fsa` has non-zero scores and inherits attributes
          for `lattice`.
        """
        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        if hasattr(lattice, "aux_labels"):
            # delete token IDs as it is not needed
            del word_fsa.aux_labels

        word_fsa.scores.zero_()
        word_fsa_with_epsilon_loops = k2.remove_epsilon_and_add_self_loops(
            word_fsa
        )

        path_to_utt_map = self.shape.row_ids(1)

        if hasattr(lattice, "aux_labels"):
            # lattice has token IDs as labels and word IDs as aux_labels.
            # inv_lattice has word IDs as labels and token IDs as aux_labels
            inv_lattice = k2.invert(lattice)
            inv_lattice = k2.arc_sort(inv_lattice)
        else:
            inv_lattice = k2.arc_sort(lattice)

        if inv_lattice.shape[0] == 1:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=torch.zeros_like(path_to_utt_map),
                sorted_match_a=True,
            )
        else:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=path_to_utt_map,
                sorted_match_a=True,
            )

        # path_lattice has word IDs as labels and token IDs as aux_labels
        path_lattice = k2.top_sort(k2.connect(path_lattice))

        one_best = k2.shortest_path(
            path_lattice, use_double_scores=use_double_scores
        )

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def compute_am_scores(self) -> k2.RaggedTensor:
        """Compute AM scores of each linear FSA (i.e., each path within
        an utterance).
        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).
        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.
        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]
        am_scores = self.fsa.scores - self.fsa.lm_scores
        ragged_am_scores = k2.RaggedTensor(scores_shape, am_scores.contiguous())
        tot_scores = ragged_am_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def compute_lm_scores(self) -> k2.RaggedTensor:
        """Compute LM scores of each linear FSA (i.e., each path within
        an utterance).
        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).
        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.
        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_lm_scores = k2.RaggedTensor(
            scores_shape, self.fsa.lm_scores.contiguous()
        )

        tot_scores = ragged_lm_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of FSAs in this Nbest.
        Note:
          Since FSAs in Nbest are just linear FSAs, log-semiring
          and tropical semiring produce the same total scores.
        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_scores = k2.RaggedTensor(
            scores_shape, self.fsa.scores.contiguous()
        )

        tot_scores = ragged_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def build_levenshtein_graphs(self) -> k2.Fsa:
        """Return an FsaVec with axes [utt][state][arc]."""
        word_ids = get_texts(self.fsa, return_ragged=True)
        return k2.levenshtein_graph(word_ids)


def one_best_decoding(
    lattice: k2.Fsa,
    use_double_scores: bool = True,
) -> k2.Fsa:
    """Get the best path from a lattice.
    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
    Return:
      An FsaVec containing linear paths.
    """
    best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
    return best_path


def nbest_decoding(
    lattice: k2.Fsa,
    num_paths: int,
    use_double_scores: bool = True,
    nbest_scale: float = 1.0,
) -> k2.Fsa:
    """It implements something like CTC prefix beam search using n-best lists.
    The basic idea is to first extract `num_paths` paths from the given lattice,
    build a word sequence from these paths, and compute the total scores
    of the word sequence in the tropical semiring. The one with the max score
    is used as the decoding output.
    Caution:
      Don't be confused by `best` in the name `n-best`. Paths are selected
      **randomly**, not by ranking their scores.
    Hint:
      This decoding method is for demonstration only and it does
      not produce a lower WER than :func:`one_best_decoding`.
    Args:
      lattice:
        The decoding lattice, e.g., can be the return value of
        :func:`get_lattice`. It has 3 axes [utt][state][arc].
      num_paths:
        It specifies the size `n` in n-best. Note: Paths are selected randomly
        and those containing identical word sequences are removed and only one
        of them is kept.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      nbest_scale:
        It's the scale applied to the `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
    Returns:
      An FsaVec containing **linear** FSAs. It axes are [utt][state][arc].
    """
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # nbest.fsa.scores contains 0s

    nbest = nbest.intersect(lattice)
    # now nbest.fsa.scores gets assigned

    # max_indexes contains the indexes for the path with the maximum score
    # within an utterance.
    max_indexes = nbest.tot_scores().argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)
    return best_path



class k2Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = k2Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech = np.expand_dims(audio, 0) # shape: [batch_size, speech_length]
        >>> speech_lengths = np.array([audio.shape[0]]) # shape: [batch_size]
        >>> batch = {"speech": speech, "speech_lengths", speech_lengths}
        >>> speech2text(batch)
        [(text, token, token_int, score), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 8,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        search_beam_size: int = 20,
        output_beam_size: int = 20,
        min_active_states: int = 30,
        max_active_states: int = 10000,
        blank_bias: float = 0.0,
        lattice_weight: float = 1.0,
        is_ctc_decoding: bool = True,
        lang_dir: Optional[str] = None,
        use_fgram_rescoring: bool = False,
        use_nbest_rescoring: bool = False,
        am_weight: float = 1.0,
        decoder_weight: float = 0.5,
        nnlm_weight: float = 1.0,
        num_paths: int = 1000,
        nbest_batch_size: int = 500,
        nll_batch_size: int = 100,
        decode_graph_config: dict = None,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        token_list = asr_model.token_list

        # save token_list
        # token_list_path = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm3/tokens.txt"
        # with open(token_list_path, "w") as fout:
        #     for tid, token in enumerate(token_list):
        #         print(f"{token} {tid}", file=fout)

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            self.lm = lm

        self.is_ctc_decoding = is_ctc_decoding
        self.use_fgram_rescoring = use_fgram_rescoring
        self.use_nbest_rescoring = use_nbest_rescoring

        assert self.is_ctc_decoding, "Currently, only ctc_decoding graph is supported."
        if self.is_ctc_decoding:
            # # self.decode_graph = k2.arc_sort(
            # #     build_ctc_topo(list(range(len(token_list))))
            # # )
            # shoud_load = True
            # self.my_lm_weight = 0.2
            # self.my_am_weight = 1 / self.my_lm_weight
            # blank_bias = 0
            # # search_beam_size = 40
            # # output_beam_size = 40
            # # max_active_states = 10000 * 10
            # # TP_path = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm3/TP.k2.fsa"
            # TP_path = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm3/TP.3.k2.fsa"
            # logging.info("self.my_lm_weight: %f" % self.my_lm_weight)
            # if shoud_load:
            #     logging.info(f"Loading graph from: " + TP_path)
            #     self.decode_graph = k2.Fsa.from_dict(
            #         torch.load(TP_path, map_location=device)
            #     )
            #     # self.decode_graph.scores *= self.my_lm_weight
            #     logging.info(f"self.decode_graph #states #arcs: {self.decode_graph.shape[0]} {self.decode_graph.num_arcs}")
            # else:
            #     # bpe_lm = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm/P_2.fst.txt"
            #     # bpe_sym_table = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm/isymb.txt"
                
            #     # bpe_lm = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm2/eval2000_small_P_2.fst.txt"
            #     # bpe_sym_table = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm2/isymb.txt"
                
            #     bpe_lm = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm3/P_3.fst.txt"
            #     bpe_sym_table = "/export/fs04/a12/rhuang/espnet/egs2/swbd/asr12//data/token_list/bpe_unigram2000/lm3/isymb.txt"
                
            #     T = k2.arc_sort(build_ctc_topo(list(range(len(token_list)))))
            #     logging.info(f"T #states #arcs: {T.shape[0]} {T.num_arcs}")
                
            #     with open(bpe_lm, "r") as f:
            #         P = k2.Fsa.from_openfst(f.read(), acceptor=False)
            #     P.labels_sym = k2.SymbolTable.from_file(bpe_sym_table)
            #     P.aux_labels_sym = k2.SymbolTable.from_file(bpe_sym_table)

            #     logging.info(f"P #states #arcs: {P.shape[0]} {P.num_arcs}")
                
            #     logging.info("Doing graph composition ...")
            #     self.decode_graph = k2.compose(T, P)
            #     self.decode_graph = k2.connect(self.decode_graph)
            #     self.decode_graph = k2.arc_sort(self.decode_graph)
            #     logging.info(f"self.decode_graph #states #arcs: {self.decode_graph.shape[0]} {self.decode_graph.num_arcs}")

            #     torch.save(self.decode_graph.as_dict(), TP_path)
            #     logging.info(f"Saved graph to: " + TP_path)

            self.decode_graph = self.load_or_build_decode_graph(decode_graph_config, token_list, device)

        self.decode_graph = self.decode_graph.to(device)
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")
        logging.info(f"Running on : {device}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.search_beam_size = search_beam_size
        self.output_beam_size = output_beam_size
        self.min_active_states = min_active_states
        self.max_active_states = max_active_states
        self.blank_bias = blank_bias
        self.lattice_weight = lattice_weight
        self.am_weight = am_weight
        self.decoder_weight = decoder_weight
        self.nnlm_weight = nnlm_weight
        self.num_paths = num_paths
        self.nbest = nbest
        self.nbest_batch_size = nbest_batch_size
        self.nll_batch_size = nll_batch_size

    
    def load_or_build_decode_graph(
        self,
        config: dict,
        token_list: List[str],
        device: str = "cpu",
    ) -> k2.Fsa:
        logging.info(f"Decode graph params : {config}")  # vars(config)

        if config.get("graph", None) is not None and Path(config["graph"]).is_file():
            logging.info(f"Loading graph from: " + config["graph"])
            decode_graph = k2.Fsa.from_dict(
                torch.load(config["graph"], map_location=device)
            )
        else:
            # 1. T graph
            T = k2.arc_sort(build_ctc_topo(list(range(len(token_list)))))
            logging.info(f"T #states #arcs : {T.shape[0]} {T.num_arcs}")

            # 2. P graph
            if config.get("P", None) is not None:
                with open(config["P"], "r") as f:
                    P = k2.Fsa.from_openfst(f.read(), acceptor=False)
                    del P.aux_labels
                    P.labels[P.labels == 2000] = 0   # handling the #0 symbol
                    P.labels[P.labels == 1999 ] = 0  # handling <sos/eos> symbol
                    P.__dict__["_properties"] = None
                    P = k2.remove_epsilon(P)   # TODO: uncomment these three lines will cause some errors in k2
                    P = k2.connect(P)
                    P = k2.arc_sort(P)
                P.labels_sym = k2.SymbolTable.from_file(config["P_symtab"])
                # P.aux_labels_sym = k2.SymbolTable.from_file(config["P_symtab"])
                logging.info(f"P #states #arcs : {P.shape[0]} {P.num_arcs}")
            else:
                P = None

            num_tokens = 2000
            num_words = 67278

            # 3. L graph for TL
            if config.get("L", None) is not None and config.get("G", None) is None:
                L = k2.Fsa.from_dict(torch.load(f'{config["L"]}'))
                del L.aux_labels
                L = k2.determinize(L)  # This will make L's labels' type change from tensors to ragged_tensors, which is not supported in later operation
                L = k2.connect(L)

                L.labels[L.labels >= num_tokens] = 0 # last symb is at index=num_tokens. Disambig symbol is the next one
                L.labels[L.labels == 1999] = 0  # handling <sos/eos> symbol
                # L.aux_labels[L.aux_labels >= num_words] = 0
                L.__dict__["_properties"] = None

                # L.aux_labels_sym = k2.SymbolTable.from_file(config["L_symtab"])

                L = k2.remove_epsilon(L)
                L = k2.connect(L)
                L = k2.arc_sort(L)
                logging.info(f"L #states #arcs : {L.shape[0]} {L.num_arcs}")
            else:
                L = None
            
            # 4. L&G graph for TLG
            if False and config.get("L", None) is not None and config.get("G", None) is not None:
                L = k2.Fsa.from_dict(torch.load(f'{config["L"]}'))
                # L = k2.determinize(L)  # This will make L's labels' type change from tensors to ragged_tensors, which is not supported in later operation
                L = k2.connect(L)
                L.aux_labels[L.aux_labels >= num_words] = 0
                L.__dict__["_properties"] = None
                L = k2.arc_sort(L)
                logging.info(f"L #states #arcs : {L.shape[0]} {L.num_arcs}")

                with open(config["G"], "r") as f:
                    G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                    del G.aux_labels
                    num_words = 67278
                    G.labels[G.labels == num_words] = 0   # handling the #0 symbol
                    # G.labels[G.labels == 1999 ] = 0  # handling <sos/eos> symbol
                    G.__dict__["_properties"] = None
                    G = k2.remove_epsilon(G)
                    G = k2.connect(G)
                    G = k2.arc_sort(G)
                # G.labels_sym = k2.SymbolTable.from_file(config["G_symtab"])
                # P.aux_labels_sym = k2.SymbolTable.from_file(config["P_symtab"])
                logging.info(f"G #states #arcs : {G.shape[0]} {G.num_arcs}")

                LG = k2.compose(L, G)
                LG = k2.connect(LG)
                LG = k2.determinize(LG)
                # Remove all necessary symbols
                LG.__dict__["_properties"] = None
                # LG.labels[LG.labels >= self.vocab_size] = 0
                # LG.aux_labels.values[LG.aux_labels.values >= osyms['#0']] = 0
                LG = k2.connect(LG)
                LG = k2.remove_epsilon(LG)
                LG = k2.arc_sort(LG)
                logging.info(f"LG #states #arcs : {LG.shape[0]} {LG.num_arcs}")

                LG.labels[LG.labels >= num_tokens] = 0 # last symb is at index=num_tokens. Disambig symbol is the next one
                LG.labels[LG.labels == 1999] = 0  # handling <sos/eos> symbol
                LG.__dict__["_properties"] = None

                LG = k2.connect(LG)
                LG = k2.remove_epsilon(LG)
                LG = k2.arc_sort(LG)
                logging.info(f"LG' #states #arcs : {LG.shape[0]} {LG.num_arcs}")
            else:
                LG = None
            
            # 4. L&G graph for TLG
            if config.get("L", None) is not None and config.get("G", None) is not None:
                L = k2.Fsa.from_dict(torch.load(f'{config["L"]}'))
                with open(config["G"], "r") as f:
                    G = k2.Fsa.from_openfst(f.read(), acceptor=False)

                L = k2.arc_sort(L)
                G = k2.arc_sort(G)

                # Attach a new attribute `lm_scores` so that we can recover
                # the `am_scores` later.
                # The scores on an arc consists of two parts:
                #  scores = am_scores + lm_scores
                # NOTE: we assume that both kinds of scores are in log-space.
                G.lm_scores = G.scores.clone()

                logging.info("Intersecting L and G")
                LG = k2.compose(L, G)
                logging.info(f"LG shape: {LG.shape}")

                logging.info("Connecting LG")
                LG = k2.connect(LG)
                logging.info(f"LG shape after k2.connect: {LG.shape}")

                logging.info(type(LG.aux_labels))
                logging.info("Determinizing LG")
                LG = k2.determinize(LG)
                logging.info(type(LG.aux_labels))

                logging.info("Connecting LG after k2.determinize")
                LG = k2.connect(LG)

                logging.info("Removing disambiguation symbols on LG")
                LG.labels[LG.labels >= num_tokens] = 0  # last symb is at index=num_tokens. Disambig symbol is the next one
                LG.labels[LG.labels == 1999] = 0  # handling <sos/eos> symbol
                # See https://github.com/k2-fsa/k2/issues/874
                # for why we need to set LG.properties to None
                LG.__dict__["_properties"] = None

                # assert isinstance(LG.aux_labels, k2.RaggedTensor)
                # LG.aux_labels.values[LG.aux_labels.values >= num_words] = 0
                del LG.aux_labels

                LG = k2.remove_epsilon(LG)
                logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

                LG = k2.connect(LG)
                # LG.aux_labels = LG.aux_labels.remove_values_eq(0)

                logging.info("Arc sorting LG")
                LG = k2.arc_sort(LG)

            else:
                LG = None

            logging.info("Doing graph composition ...")
            if P is not None:
                logging.info("Getting TP decoding graph...")
                decode_graph = k2.compose(T, P)
                decode_graph = k2.connect(decode_graph)
                decode_graph = k2.arc_sort(decode_graph)
            elif L is not None and LG is None:
                logging.info("Getting TL decoding graph...")
                decode_graph = k2.compose(T, L)
                decode_graph = k2.connect(decode_graph)
                decode_graph = k2.arc_sort(decode_graph)
                logging.info(f"TL #states #arcs : {decode_graph.shape[0]} {decode_graph.num_arcs}")
            elif LG is not None:
                logging.info("Getting TLG decoding graph...")
                # LG = k2.compose(L, G)
                # LG = k2.connect(LG)
                # LG = k2.determinize(LG)
                # # Remove all necessary symbols
                # LG.__dict__["_properties"] = None
                # # LG.labels[LG.labels >= self.vocab_size] = 0
                # # LG.aux_labels.values[LG.aux_labels.values >= osyms['#0']] = 0
                # LG = k2.connect(LG)
                # LG = k2.remove_epsilon(LG)
                # LG = k2.arc_sort(LG)
                # logging.info(f"LG #states #arcs : {LG.shape[0]} {LG.num_arcs}")
                decode_graph = k2.compose(T, LG)
                # decode_graph = k2.compose(decode_graph, G)
                decode_graph = k2.connect(decode_graph)
                decode_graph = k2.arc_sort(decode_graph)
                logging.info(f"TLG #states #arcs : {decode_graph.shape[0]} {decode_graph.num_arcs}")
            else:
                decode_graph = T
            
            if config.get("graph_save_path", None) is not None and not Path(config["graph_save_path"]).is_file():
                torch.save(decode_graph.as_dict(), config["graph_save_path"])
                logging.info(f"Saved the decoding graph to : " + config["graph_save_path"])

        logging.info(f"Decode_graph #states #arcs : {decode_graph.shape[0]} {decode_graph.num_arcs}")

        decode_graph.scores *= config.get("lm_weight", 1)
        return decode_graph


    @torch.no_grad()
    def __call__(
        self, batch: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> List[Tuple[Optional[str], List[str], List[int], float]]:
        """Inference

        Args:
            batch: Input speech data and corresponding lengths
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        if isinstance(batch["speech"], np.ndarray):
            batch["speech"] = torch.tensor(batch["speech"])
        if isinstance(batch["speech_lengths"], np.ndarray):
            batch["speech_lengths"] = torch.tensor(batch["speech_lengths"])

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        # enc: [N, T, C]
        enc, encoder_out_lens = self.asr_model.encode(**batch)

        # logp_encoder_output: [N, T, C]
        logp_encoder_output = torch.nn.functional.log_softmax(
            self.asr_model.ctc.ctc_lo(enc), dim=2
        )

        # It maybe useful to tune blank_bias.
        # The valid range of blank_bias is [-inf, 0]
        logp_encoder_output[:, :, 0] += self.blank_bias

        batch_size = encoder_out_lens.size(0)
        sequence_idx = torch.arange(0, batch_size).unsqueeze(0).t().to(torch.int32)
        start_frame = torch.zeros([batch_size], dtype=torch.int32).unsqueeze(0).t()
        num_frames = encoder_out_lens.cpu().unsqueeze(0).t().to(torch.int32)
        supervision_segments = torch.cat([sequence_idx, start_frame, num_frames], dim=1)

        supervision_segments = supervision_segments.to(torch.int32)

        # An introduction to DenseFsaVec:
        # https://k2-fsa.github.io/k2/core_concepts/index.html#dense-fsa-vector
        # It could be viewed as a fsa-type lopg_encoder_output,
        # whose weight on the arcs are initialized with logp_encoder_output.
        # The goal of converting tensor-type to fsa-type is using
        # fsa related functions in k2. e.g. k2.intersect_dense_pruned below
        dense_fsa_vec = k2.DenseFsaVec(logp_encoder_output, supervision_segments)
        # dense_fsa_vec = k2.DenseFsaVec(logp_encoder_output * self.my_am_weight, supervision_segments)

        # The term "intersect" is similar to "compose" in k2.
        # The differences is are:
        # for "compose" functions, the composition involves
        # mathcing output label of a.fsa and input label of b.fsa
        # while for "intersect" functions, the composition involves
        # matching input label of a.fsa and input label of b.fsa
        # Actually, in compose functions, b.fsa is inverted and then
        # a.fsa and inv_b.fsa are intersected together.
        # For difference between compose and interset:
        # https://github.com/k2-fsa/k2/blob/master/k2/python/k2/fsa_algo.py#L308
        # For definition of k2.intersect_dense_pruned:
        # https://github.com/k2-fsa/k2/blob/master/k2/python/k2/autograd.py#L648
        lattices = k2.intersect_dense_pruned(
            self.decode_graph,
            dense_fsa_vec,
            self.search_beam_size,
            self.output_beam_size,
            self.min_active_states,
            self.max_active_states,
        )

        # lattices.scores is the sum of decode_graph.scores(a.k.a. lm weight) and
        # dense_fsa_vec.scores(a.k.a. am weight) on related arcs.
        # For ctc decoding graph, lattices.scores only store am weight
        # since the decoder_graph only define the ctc topology and
        # has no lm weight on its arcs.
        # While for 3-gram decoding, whose graph is converted from language models,
        # lattice.scores contains both am weights and lm weights
        #
        # It maybe useful to tune lattice.scores
        # The valid range of lattice_weight is [0, inf)
        # The lattice_weight will affect the search of k2.random_paths
        lattices.scores *= self.lattice_weight

        results = []
        if self.use_nbest_rescoring:
            (
                am_scores,
                lm_scores,
                token_ids,
                new2old,
                path_to_seq_map,
                seq_to_path_splits,
            ) = nbest_am_lm_scores(
                lattices, self.num_paths, self.device, self.nbest_batch_size
            )

            ys_pad_lens = torch.tensor([len(hyp) for hyp in token_ids]).to(self.device)
            max_token_length = max(ys_pad_lens)
            ys_pad_list = []
            for hyp in token_ids:
                ys_pad_list.append(
                    torch.cat(
                        [
                            torch.tensor(hyp, dtype=torch.long),
                            torch.tensor(
                                [self.asr_model.ignore_id]
                                * (max_token_length.item() - len(hyp)),
                                dtype=torch.long,
                            ),
                        ]
                    )
                )

            ys_pad = (
                torch.stack(ys_pad_list).to(torch.long).to(self.device)
            )  # [batch, max_token_length]

            encoder_out = enc.index_select(0, path_to_seq_map.to(torch.long)).to(
                self.device
            )  # [batch, T, dim]
            encoder_out_lens = encoder_out_lens.index_select(
                0, path_to_seq_map.to(torch.long)
            ).to(
                self.device
            )  # [batch]

            decoder_scores = -self.asr_model.batchify_nll(
                encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, self.nll_batch_size
            )

            # padded_value for nnlm is 0
            ys_pad[ys_pad == self.asr_model.ignore_id] = 0
            nnlm_nll, x_lengths = self.lm.batchify_nll(
                ys_pad, ys_pad_lens, self.nll_batch_size
            )
            nnlm_scores = -nnlm_nll.sum(dim=1)

            batch_tot_scores = (
                self.am_weight * am_scores
                + self.decoder_weight * decoder_scores
                + self.nnlm_weight * nnlm_scores
            )
            split_size = indices_to_split_size(
                seq_to_path_splits.tolist(), total_elements=batch_tot_scores.size(0)
            )
            batch_tot_scores = torch.split(
                batch_tot_scores,
                split_size,
            )

            hyps = []
            scores = []
            processed_seqs = 0
            for tot_scores in batch_tot_scores:
                if tot_scores.nelement() == 0:
                    # the last element by torch.tensor_split may be empty
                    # e.g.
                    # torch.tensor_split(torch.tensor([1,2,3,4]), torch.tensor([2,4]))
                    # (tensor([1, 2]), tensor([3, 4]), tensor([], dtype=torch.int64))
                    break
                best_seq_idx = processed_seqs + torch.argmax(tot_scores)

                assert best_seq_idx < len(token_ids)
                best_token_seqs = token_ids[best_seq_idx]
                processed_seqs += tot_scores.nelement()
                hyps.append(best_token_seqs)
                scores.append(tot_scores.max().item())

            assert len(hyps) == len(split_size)
        # else:
        #     best_paths = k2.shortest_path(lattices, use_double_scores=True)
        #     scores = best_paths.get_tot_scores(
        #         use_double_scores=True, log_semiring=False
        #     ).tolist()
        #     hyps = get_texts(best_paths)
        else:
            nbest = Nbest.from_lattice(
                lattice=lattices,
                num_paths=max(self.num_paths, self.nbest * 10),
                use_double_scores=True,
                nbest_scale=0.8,
            )

            nbest2 = nbest.intersect(lattices)
            tot_scores = nbest2.tot_scores()

            topk = tot_scores.tolist()[0]
            topk = torch.as_tensor(topk).topk(min(self.num_paths, len(topk)))
            topk_scores = topk.values.tolist()
            topk_indices = topk.indices.type(torch.int32)
            best_paths = k2.index_fsa(nbest2.fsa, topk_indices)

            scores = topk_scores
            hyps = get_texts(best_paths)

            # best_paths = k2.shortest_path(lattices, use_double_scores=True)
            # scores = best_paths.get_tot_scores(
            #     use_double_scores=True, log_semiring=False
            # ).tolist()
            # hyps = get_texts(best_paths)

        assert len(scores) == len(hyps)

        for token_int, score in zip(hyps, scores):
            # For decoding methods nbest_rescoring and ctc_decoding
            # hyps stores token_index, which is lattice.labels.

            # convert token_id to text with self.tokenizer
            token = self.converter.ids2tokens(token_int)
            assert self.tokenizer is not None
            text = self.tokenizer.tokens2text(token)
            results.append((text, token, token_int, score))

        best_path_aux_labels = best_paths[0].aux_labels.tolist()
        # best_path_aux_labels = self.converter.ids2tokens(best_path_aux_labels)
        best_path_aux_labels =  [str(lbl) for lbl in best_path_aux_labels]

        assert check_return_type(results)
        # return [results[0]]
        return results, best_path_aux_labels

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build k2Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return k2Speech2Text(**kwargs)


def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    streaming: bool,
    is_ctc_decoding: bool,
    use_nbest_rescoring: bool,
    num_paths: int,
    nbest_batch_size: int,
    nll_batch_size: int,
    k2_config: Optional[str],
):
    assert is_ctc_decoding, "Currently, only ctc_decoding graph is supported."
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)
    with open(k2_config) as k2_config_file:
        dict_k2_config = yaml.safe_load(k2_config_file)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        is_ctc_decoding=is_ctc_decoding,
        use_nbest_rescoring=use_nbest_rescoring,
        num_paths=num_paths,
        nbest_batch_size=nbest_batch_size,
        nll_batch_size=nll_batch_size,
    )

    speech2text_kwargs = dict(**speech2text_kwargs, **dict_k2_config)
    speech2text = k2Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    with DatadirWriter(output_dir) as writer:
        start_decoding_time = datetime.datetime.now()
        for batch_idx, (keys, batch) in enumerate(loader):
            if batch_idx % 10 == 0:
                logging.info(f"Processing {batch_idx} batch")
                # if batch_idx > 20: break
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            # 1-best list of (text, token, token_int)
            results, best_path_aux_labels = speech2text(batch)

            # for key_idx, (text, token, token_int, score) in enumerate(results):
            #     key = keys[key_idx]
            #     best_writer = writer["1best_recog"]
            #     # Write the result to each file
            #     best_writer["token"][key] = " ".join(token)
            #     best_writer["token_int"][key] = " ".join(map(str, token_int))
            #     best_writer["score"][key] = str(score)

            #     if text is not None:
            #         best_writer["text"][key] = text

            # Only supporting batch_size==1
            key = keys[0]
            for n, (text, token, token_int, score) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(score)

                if text is not None:
                    ibest_writer["text"][key] = text
            
            best_writer = writer["1best_recog"]
            best_writer["best_path_aux_labels"][key] = " ".join(best_path_aux_labels)

        end_decoding_time = datetime.datetime.now()
        decoding_duration = end_decoding_time - start_decoding_time
        logging.info(f"Decoding duration is {decoding_duration.seconds} seconds")


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--streaming", type=str2bool, default=False)

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--is_ctc_decoding",
        type=str2bool,
        default=True,
        help="Use ctc topology as decoding graph",
    )
    group.add_argument("--use_nbest_rescoring", type=str2bool, default=False)
    group.add_argument(
        "--num_paths",
        type=int,
        default=1000,
        help="The third argument for k2.random_paths",
    )
    group.add_argument(
        "--nbest_batch_size",
        type=int,
        default=500,
        help="batchify nbest list when computing am/lm scores to avoid OOM",
    )
    group.add_argument(
        "--nll_batch_size",
        type=int,
        default=100,
        help="batch_size when computing nll during nbest rescoring",
    )
    group.add_argument("--k2_config", type=str, help="Config file for decoding with k2")

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
