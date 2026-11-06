"""Each vocabulary part -> the real-life objects (or object parts) LEGO builders use it to depict, by a local VLM
looking at its four renders and its name.  -> <out>/objects.jsonl, one {"ldraw", "name", "objects"} per line, resumable."""
import sys, json, base64, io, time, urllib.request; sys.path[:0]=["src","tools","tools/set_study"]
import numpy as np
from pathlib import Path
from PIL import Image
from part_embed import part_images
OUT=Path(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 10**9; MODEL=sys.argv[3] if len(sys.argv)>3 else "qwen3.5:9b"
E=np.load("/Volumes/PRO-G40/lego_data/analysis/part_embed/google__siglip2-so400m-patch14-384.npz")
f=OUT/"objects.jsonl"; done={json.loads(l)["ldraw"] for l in open(f)} if f.exists() else set()
PROMPT=("This is the LEGO part \"{name}\" (category {cat}), drawn from four sides. In LEGO sets and fan models, which "
        "real-life objects or parts of objects is it used to depict? Think of buildings, landscapes, vehicles, "
        "furniture, nature. Answer with JSON only: {{\"objects\": [8 to 12 short noun phrases, most typical first]}}")
def grid(n):
    g=Image.new("RGB",(448,448),"white")
    for k,im in enumerate(part_images(n,224)[:4]): g.paste(im,(224*(k%2),224*(k//2)))
    b=io.BytesIO(); g.save(b,"PNG"); return base64.b64encode(b.getvalue()).decode()
order=[i for i in np.argsort(E["ldraw"]) if E["ldraw"][i] not in done][:N]
with open(f,"a") as out:
    for i in order:
        t=time.time(); n,nm,c=str(E["ldraw"][i]),str(E["name"][i]),str(E["category"][i])
        req={"model":MODEL,"stream":False,"think":False,"format":"json","options":{"temperature":0},
             "messages":[{"role":"user","content":PROMPT.format(name=nm,cat=c),"images":[grid(n)]}]}
        r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat",json.dumps(req).encode(),{"Content-Type":"application/json"}),timeout=600))
        try: objs=[str(o).strip().lower() for o in json.loads(r["message"]["content"])["objects"]][:12]
        except Exception: objs=[]
        out.write(json.dumps({"ldraw":n,"name":nm,"objects":objs})+"\n"); out.flush()
        print(f"{time.time()-t:5.1f}s {nm[:45]:45s} -> {', '.join(objs)}",flush=True)
