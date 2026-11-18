"""V4: generate the CC^L-style background vocabulary (Wysoczanska et al., TMLR 2025, App. D.2 prompt) with Qwen2.5-7B-Instruct,
greedy decoding; union over each dataset's foreground classes; drop words whose CLIP ViT-B/16 text similarity to any
foreground class name exceeds 0.9 (beta = 0.9 as in the paper). Usage: python p4_gen_vocab.py MODEL_DIR OUT_DIR"""

import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, OUT = sys.argv[1:3]
PROMPT = "\n".join([
    "You are a helpful AI assistant with visual abilities.",
    "Given an input object O, I want you to generate a list of words related to objects that can be surrounding input object O in an image to help me perform semantic segmentation.",
    "If the input object is 'fork', you can generate a list of words such as [\"bottle\", \"knife\", \"table\", \"napkin\", \"bread\"]",
    "If the input object is 'child', you can generate a list of words such as [\"toy\", \"drawing\", \"bed\", \"room\", \"playground\"]",
    "You should not generate synonyms of input object O, nor parts of input object O.",
    "Generate a list of objects surrounding the input object {q} without any synonym nor parts, nor content of it.",
    "Answer with a list of words. No explanation.",
    "Answer:"])
NAMES = "/root/autodl-tmp/p2/repos/ClearCLIP/configs/cls_{}.txt"


def parse(text):
    m = re.search(r"\[(.*?)\]", text, flags=re.S)
    body = m.group(1) if m else text
    items = re.split(r"[,\n]", body)
    out = []
    for it in items:
        w = re.sub(r"^[\s\-\*\d\.\)]+", "", it).strip().strip("\"'" + chr(96)).strip().lower()
        if w and w not in out:
            out.append(w)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    raw, vocab = {}, {}
    sys.path.insert(0, "/root/autodl-tmp/p2/repos/SCLIP")
    import clip  # SCLIP's bundled OpenAI CLIP
    cm, _ = clip.load("ViT-B/16", device="cuda", jit=False)

    def emb(words):
        with torch.no_grad():
            f = cm.encode_text(clip.tokenize([f"a photo of a {w}." for w in words]).cuda()).float()
        return f / f.norm(dim=-1, keepdim=True)

    for ds in ["voc21", "coco_object"]:
        classes = [l.strip() for l in open(NAMES.format(ds)).read().split("\n") if l.strip()][1:]
        raw[ds] = {}
        words = []
        for c in classes:
            msgs = [{"role": "user", "content": PROMPT.replace("{q}", c)}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=128, do_sample=False)
            text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
            raw[ds][c] = text
            for w in parse(text):
                if w not in words:
                    words.append(w)
        words = [w for w in words if w != "background"]
        sim = emb(words) @ emb(classes).T
        keep = [w for w, s in zip(words, sim.max(1).values.tolist()) if s <= 0.9]
        vocab[ds] = keep
        print(ds, "classes", len(classes), "generated", len(words), "kept", len(keep), "examples", keep[:15], flush=True)
    json.dump(raw, open(f"{OUT}/cc_raw_generations.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(vocab, open(f"{OUT}/bg_words_cc.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
