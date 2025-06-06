"""
Bake the chosen refusal-ablating direction into a Qwen-2.5 model.
Run after you have produced `direction.pt` with select_direction.py.
$ python bake_direction.py --model Qwen/Qwen2.5-72B-Instruct \
                           --direction direction.pt \
                           --out_path qwen2.5-72B-no-refusal
"""
import argparse, torch

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM
from pipeline.model_utils.qwen3_model import orthogonalize_qwen3_weights



def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    direction = torch.load(args.direction)            # 1 × d_model
    direction = direction / direction.norm()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),# device_map="auto",
        trust_remote_code=True,# attn_implementation="flash_attention_2"
    )
    model.eval(); model.requires_grad_(False)

    orthogonalize_qwen3_weights(model, direction)

    print("Saving…"); model.save_pretrained(args.out_path)
    print("Done.  Load with AutoModelForCausalLM.from_pretrained(...)")

if __name__ == "__main__":
    main()
