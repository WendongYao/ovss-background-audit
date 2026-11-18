"""V5: conflict-filtered background vocabulary CF (CONTRACT). Context-60 CC words generated exactly as in V4
(p4_gen_vocab: same prompt, Qwen2.5-7B-Instruct greedy, beta = 0.9 CLIP filter); VOC21 / COCO-Object start from V4's
bg_words_cc.json. Then: drop non-ASCII words; drop words an LLM judge (fixed prompt, greedy) labels as a category,
synonym, subtype or part of any foreground class. Usage: python p5_vocab.py MODEL_DIR V4_CC_JSON OUT_DIR"""

import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, V4CC, OUT = sys.argv[1:4]
NAMES = "/root/autodl-tmp/p2/repos/ClearCLIP/configs/cls_{}.txt"
GEN_PROMPT = "\n".join([
    "You are a helpful AI assistant with visual abilities.",
    "Given an input object O, I want you to generate a list of words related to objects that can be surrounding input object O in an image to help me perform semantic segmentation.",
    "If the input object is 'fork', you can generate a list of words such as [\"bottle\", \"knife\", \"table\", \"napkin\", \"bread\"]",
    "If the input object is 'child', you can generate a list of words such as [\"toy\", \"drawing\", \"bed\", \"room\", \"playground\"]",
    "You should not generate synonyms of input object O, nor parts of input object O.",
    "Generate a list of objects surrounding the input object {q} without any synonym nor parts, nor content of it.",
    "Answer with a list of words. No explanation.",
    "Answer:"])
JUDGE_PROMPT = "\n".join([
    "You are given a word W and a list of object categories. Answer yes if W names one of the categories, a synonym of one, a specific kind (subtype) of one, or a part of one. Otherwise answer no.",
    "W: {w}",
    "Categories: {cats}",
    "Answer with yes or no."])


def parse(text):
    m = re.search(r"\[(.*?)\]", text, flags=re.S)
    body = m.group(1) if m else text
    out = []
    for it in re.split(r"[,\n]", body):
        w = re.sub(r"^[\s\-\*\d\.\)]+", "", it).strip().strip("\"'" + chr(96)).strip().lower()
        if w and w not in out:
            out.append(w)
    return out


def classes_of(ds):
    return [l.strip() for l in open(NAMES.format(ds), encoding="utf-8").read().split("\n") if l.strip()][1:]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

    def chat(text, max_new):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

    sys.path.insert(0, "/root/autodl-tmp/p2/repos/SCLIP")
    import clip
    cm, _ = clip.load("ViT-B/16", device="cuda", jit=False)

    def emb(words):
        with torch.no_grad():
            f = cm.encode_text(clip.tokenize([f"a photo of a {w}." for w in words]).cuda()).float()
        return f / f.norm(dim=-1, keepdim=True)

    cc = json.load(open(V4CC, encoding="utf-8"))
    raw = {"context60": {}}
    words = []
    for c in classes_of("context60"):
        t = chat(GEN_PROMPT.replace("{q}", c), 128)
        raw["context60"][c] = t
        for w in parse(t):
            if w not in words:
                words.append(w)
    words = [w for w in words if w != "background"]
    ctx_classes = classes_of("context60")
    sim = emb(words) @ emb(ctx_classes).T
    cc["context60"] = [w for w, s in zip(words, sim.max(1).values.tolist()) if s <= 0.9]
    print("context60 CC generated", len(words), "kept", len(cc["context60"]), flush=True)

    cf, judged = {}, {}
    for ds in ["voc21", "coco_object", "context60"]:
        cats = ", ".join(classes_of(ds))
        keep, judged[ds] = [], {}
        for w in cc[ds]:
            if not w.isascii():
                judged[ds][w] = "non-ascii"
                continue
            a = chat(JUDGE_PROMPT.replace("{w}", w).replace("{cats}", cats), 4).strip().lower()
            judged[ds][w] = a
            if not a.startswith("yes"):
                keep.append(w)
        cf[ds] = keep
        removed = [w for w in cc[ds] if w not in keep]
        print(ds, "CC", len(cc[ds]), "CF", len(keep), "removed", removed, flush=True)
    json.dump(raw, open(f"{OUT}/cc_raw_generations_context60.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(cc, open(f"{OUT}/bg_words_cc3.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(judged, open(f"{OUT}/cf_judgements.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(cf, open(f"{OUT}/bg_words_cf.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
