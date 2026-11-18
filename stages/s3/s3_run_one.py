"""S3: run one cell with S2's s2_run_one.main, pointed at S3 specs and S3 collision file (no other change).
Usage: python s3_run_one.py METHOD SETTING VARIANT OUT_JSON [--smoke N]   (VARIANT: voc21_orig | coco_object_orig)"""
import sys
sys.path.insert(0, "/root/autodl-tmp/s2/code")
import s2_run_one as s2  # noqa: E402

s2.S1 = "/root/autodl-tmp/s3"   # specs/<variant>.json
s2.S2 = "/root/autodl-tmp/s3"   # s2_collisions.json -> s3_collisions.json
if __name__ == "__main__":
    if sys.argv[1] != "Trident":
        s2.lm.patch(sys.argv[1])
    s2.main()
