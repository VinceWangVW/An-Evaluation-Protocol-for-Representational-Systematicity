#!/usr/bin/env python3
"""Run a complete Binding audit over declared seven-item matched sets."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from experiments.matched_audit import audit_matched_sets,write_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--family",choices=["binding"],default="binding")
    parser.add_argument("--layer-index",type=int,default=-1)
    parser.add_argument("--max-new-tokens",type=int,default=192)
    parser.add_argument("--seed",type=int,default=13)
    parser.add_argument("--permutations",type=int,default=199)
    parser.add_argument("--realism-margin",type=float,default=.05)
    parser.add_argument("--profile",type=Path)
    parser.add_argument("--device")
    parser.add_argument("--decoder-cross-attention-scale",type=float)
    args=parser.parse_args()
    from models.adapters import build_model_adapter
    kwargs={"family":"seq2seq_lm","prompt_style":"raw",
            "decoder_cross_attention_scale":args.decoder_cross_attention_scale}
    if args.device:kwargs["device"]=args.device
    adapter=build_model_adapter(args.checkpoint,**kwargs)
    rows=[json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    profile=json.loads(args.profile.read_text()) if args.profile else None
    payload=audit_matched_sets(rows,adapter,layer_index=args.layer_index,max_new_tokens=args.max_new_tokens,
        seed=args.seed,permutations=args.permutations,profile=profile,realism_margin=args.realism_margin)
    write_audit(args.output_dir,payload)
    print(json.dumps({"status":payload["status"],"counts":payload["summary"]["counts"],
                      "output_directory":str(args.output_dir)},indent=2))


if __name__=="__main__":main()
