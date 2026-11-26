# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""Sentence splitting and markup cleaning on cases taken from the confdoc004 corpus."""
import data_loader as dl


def test_initials_do_not_split_names():
    s = dl.split_sentences("The cup was proposed by Captain James T. Sutherland during the war. It was first awarded in 1919.")
    assert s == ["The cup was proposed by Captain James T. Sutherland during the war.", "It was first awarded in 1919."]


def test_run_of_initials():
    s = dl.split_sentences("Past President Capt. J. T. Sutherland spoke. He was in France.")
    assert s == ["Past President Capt. J. T. Sutherland spoke.", "He was in France."]


def test_roman_numeral_still_ends_sentence():
    s = dl.split_sentences("He served in World War I. After the war he returned home.")
    assert s == ["He served in World War I.", "After the war he returned home."]


def test_extra_abbreviations():
    s = dl.split_sentences("He died on Feb. 3, 1920. Sgt. Brown and Lt. Smith ca. 40 men followed.")
    assert s == ["He died on Feb. 3, 1920.", "Sgt. Brown and Lt. Smith ca. 40 men followed."]


def test_existing_split_behaviour_kept():
    s = dl.split_sentences("The fossil is 30.2 million years old. It was found in Egypt! Was it a primate?")
    assert s == ["The fossil is 30.2 million years old.", "It was found in Egypt!", "Was it a primate?"]


def test_image_options_removed():
    c = dl.clean_wiki_text("Battle of St Albans upright=1.5|thumb|Map of the first Battle of St Albans.")
    assert "|" not in c and "upright" not in c and "Map of the first Battle" in c
    c = dl.clean_wiki_text('The shield shows the number "3".|left|thumb|120px|The official emblem of the battalion.')
    assert "|" not in c and "120px" not in c and "The official emblem" in c
    c = dl.clean_wiki_text("Netherlands in Asia 250px|Fort St. George was founded at Madras in 1639.")
    assert c == "Netherlands in Asia Fort St. George was founded at Madras in 1639."


def test_table_rows_not_usable():
    assert not dl._is_usable_sentence("| Pepper X | 3.18M SHU | Dragon's Breath | 2.48M SHU | Carolina Reaper", 4, 80)
    assert not dl._is_usable_sentence('!! rowspan="2" Description !! colspan="2" Example and more text', 4, 80)
    assert not dl._is_usable_sentence("Note that the body hair of both models is removed. |alt=Photograph of a woman.", 4, 80)
    assert dl._is_usable_sentence("The station opened to passengers on 1 May 1902.", 4, 80)
