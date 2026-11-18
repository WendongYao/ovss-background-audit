"""Unit tests for ovss_collision_check (run: python -m unittest test_ovss_collision_check.py)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import ovss_collision_check as cc

HERE = os.path.dirname(os.path.abspath(__file__))


class Rule(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(cc.normalise("Wall-Concrete"), "wall concrete")
        self.assertEqual(cc.normalise("sky-other"), "sky")
        self.assertEqual(cc.normalise("tree_merged"), "tree")
        self.assertEqual(cc.normalise("  potted   plant "), "potted plant")

    def test_match(self):
        self.assertTrue(cc.collide("bed", "bed"))
        self.assertTrue(cc.collide("sky", "sky-other"))           # trailing 'other' dropped
        self.assertTrue(cc.collide("wall", "wall-concrete"))      # w is a word of q
        self.assertTrue(cc.collide("dining table", "table"))      # q is a word of w
        self.assertFalse(cc.collide("sea", "seat"))               # whole words only
        self.assertFalse(cc.collide("sky", "skyscraper"))
        self.assertFalse(cc.collide("tree", "street"))

    def test_background_never_removed(self):
        r = cc.check("background, bed", ["background", "bed"])
        self.assertEqual(r["kept"], ["background"])
        self.assertEqual(r["removed"], {"bed": ["bed"]})

    def test_empty_falls_back_to_background(self):
        r = cc.check("bed, wall", ["bed", "wall-other"])
        self.assertEqual(r["clean_background_line"], "background")

    def test_queries_untouched_and_separator(self):
        r = cc.check("sky; wall; lamp", ["wall-brick; wall-stone", "person"], sep=";")
        self.assertEqual(r["clean_background_line"], "sky; lamp")
        self.assertEqual(r["removed"], {"wall": ["wall-brick", "wall-stone"]})

    def test_synonyms_inside_query_lines(self):
        r = cc.check("sky, desk, bed", ["person, person in shirt", "table, desk"])
        self.assertEqual(sorted(r["removed"]), ["desk"])


class Cli(unittest.TestCase):
    def test_example_files(self):
        ex = os.path.join(HERE, "examples")
        with tempfile.TemporaryDirectory() as d:
            out, rep = os.path.join(d, "clean.txt"), os.path.join(d, "rep.json")
            p = subprocess.run([sys.executable, os.path.join(HERE, "ovss_collision_check.py"),
                                os.path.join(ex, "toy_coco_object.txt"), "--out", out, "--json", rep, "--strict"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)  # collisions found under --strict
            r = json.load(open(rep, encoding="utf-8"))
            self.assertEqual(sorted(r["removed"]), ["bed", "wall"])
            self.assertEqual(open(out, encoding="utf-8").read().splitlines()[0], "sky, tree, lamp")
            p = subprocess.run([sys.executable, os.path.join(HERE, "ovss_collision_check.py"),
                                os.path.join(ex, "toy_coco_object.txt"), "--targets", os.path.join(ex, "toy_new_domain.txt")],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0)
            self.assertIn("'tree'", p.stdout)


if __name__ == "__main__":
    unittest.main()
