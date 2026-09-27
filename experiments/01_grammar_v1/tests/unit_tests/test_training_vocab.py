"""Tests for the grammar-selectable vocabulary (training.data.select_grammar)
and the dataset-aware run names (training.train.dataset_tag).

Grammar V1 stays the default vocabulary (111 ids: 6 specials + 105 words), so
every existing V1 run keeps its token ids; the number-agreement grammar adds
the plural nouns and base-form verbs (6 + 183 = 189 ids). Special tokens are
first in both, so <pad>/<bos>/<eos>/<sep> ids never change.
"""

import pytest
from cfg import grammar, grammar_number_agreement
from training import data
from training.data import PAD_ID, SPECIALS, STOI, VOCAB, build_vocab, encode_row, grammar_for, select_grammar
from training.train import dataset_tag


@pytest.fixture(autouse=True)
def restore_v1():
    yield
    select_grammar("v1")


def test_grammar_for_dataset_names():
    assert grammar_for("kylelovesllms/grammar-v1-frames-ce-rb") == "v1"
    assert grammar_for("kylelovesllms/grammar-number-agreement-frames-ce") == "number_agreement"
    assert grammar_for("/some/dir/grammar-number-agreement-frames-ce-rb") == "number_agreement"
    assert grammar_for("/some/dir/my-data") == "v1"


def test_v1_is_the_default_and_reselecting_it_is_a_no_op():
    v1 = list(VOCAB)
    assert data.active_grammar() == "v1" and len(v1) == 111
    assert v1 == build_vocab(grammar.LEXICON)
    assert select_grammar("v1") == 111 and list(VOCAB) == v1


def test_number_agreement_vocab_keeps_special_ids_and_covers_plurals():
    assert select_grammar("number_agreement") == 6 + 183 == 189
    assert data.active_grammar() == "number_agreement"
    assert VOCAB[: len(SPECIALS)] == list(SPECIALS) and STOI["<pad>"] == PAD_ID == 0
    assert set(VOCAB[len(SPECIALS):]) == {w for words in grammar_number_agreement.LEXICON.values() for w in words}
    ids = encode_row("translate", "cats that dog hugs tease bird", "dog hugs that cats bird tease")
    assert [VOCAB[i] for i in ids] == ["<bos>", "cats", "that", "dog", "hugs", "tease", "bird", "<translate>", "<sep>", "dog", "hugs", "that", "cats", "bird", "tease", "<eos>"]
    select_grammar("v1")
    with pytest.raises(KeyError):
        encode_row("copy", "cats swim", "cats swim")


def test_select_grammar_propagates_to_importers():
    """train.py and evaluate.py bind VOCAB/STOI at import time; the switch
    mutates those objects in place so their bindings follow."""
    from training.evaluate import EOS_ID, SEP_ID, _words

    select_grammar("number_agreement")
    assert data.VOCAB is VOCAB and data.STOI is STOI
    assert _words(encode_row("copy", "cats swim", "cats swim")) == "<bos> cats swim <copy> <sep> cats swim <eos>"
    assert VOCAB[EOS_ID] == "<eos>" and VOCAB[SEP_ID] == "<sep>"


def test_dataset_tag_names_dataset_and_keeps_v1_tags():
    assert dataset_tag("kylelovesllms/grammar-v1-frames-ce-rb") == "ce-rb"
    assert dataset_tag("kylelovesllms/grammar-number-agreement-frames-ce") == "number-agreement-ce"
    assert dataset_tag("kylelovesllms/grammar-number-agreement-frames-ce-rb") == "number-agreement-ce-rb"
    assert dataset_tag("/local/grammar-number-agreement-frames-rb") == "number-agreement-rb"
    assert dataset_tag("/some/local/dir/my-data") == "my-data"
