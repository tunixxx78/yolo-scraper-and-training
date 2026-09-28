import streamlit as st
from pathlib import Path
from PIL import Image
import requests, base64, io, json, random, shutil, re
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Qwen → YOLO Annotator v0.4.2", layout="wide")
st.title("Qwen → YOLO Annotator v0.4.2")
st.caption("Qwen3-VL auto-annotation + draggable/resizable boxes + cumulative multi-class YOLO dataset export. Original collector images are never modified.")

def parse_json(s):
    try:return json.loads(s)
    except Exception:
        m=re.search(r'\{.*\}',s,re.S)
        return json.loads(m.group(0)) if m else {}

def qwen_boxes(base,model,image,class_name):
    im=image.convert("RGB"); cp=im.copy(); cp.thumbnail((1536,1536))
    b=io.BytesIO(); cp.save(b,"JPEG",quality=88); enc=base64.b64encode(b.getvalue()).decode()
    schema={"type":"object","properties":{"objects":{"type":"array","items":{"type":"object","properties":{
        "class_name":{"type":"string"},"confidence":{"type":"number","minimum":0,"maximum":1},
        "bbox":{"type":"array","items":{"type":"number","minimum":0,"maximum":1000},"minItems":4,"maxItems":4}
    },"required":["class_name","confidence","bbox"]}}},"required":["objects"]}
    prompt=f"""Annotate every clearly visible instance of this target for an object-detection dataset:
TARGET CLASS: {class_name}
Return tight bounding boxes around physical target objects only.
Coordinates MUST be [xmin,ymin,xmax,ymax] normalized 0..1000 relative to the full image.
Do not box people, text, signs, scenery, or related equipment unless it is the requested target.
Include all visible target instances. If none are visible, return an empty objects array."""
    payload={"model":model,"messages":[{"role":"user","content":prompt,"images":[enc]}],
             "stream":False,"format":schema,"options":{"temperature":0}}
    r=requests.post(base.rstrip("/")+"/api/chat",json=payload,timeout=240);r.raise_for_status()
    d=parse_json(r.json().get("message",{}).get("content",""));out=[]
    for o in d.get("objects",[]):
        try:
            box=[max(0,min(1000,float(x))) for x in o["bbox"]];x1,y1,x2,y2=box
            if x2>x1 and y2>y1:out.append({"confidence":float(o.get("confidence",0)),"bbox":box,"source":"qwen"})
        except Exception:pass
    return out

def clean_record(r):return {k:v for k,v in r.items() if not k.startswith("_")}
def save_record(r):r["_json"].write_text(json.dumps(clean_record(r),indent=2),encoding="utf-8")

def safe_folder_name(name):
    # Windows-safe while keeping equipment names readable.
    name=re.sub(r'[<>:"/\\|?*]+',"_",str(name)).strip().rstrip(". ")
    return name or "unnamed_model"

def load_root_classes(dataset_root):
    """Read persistent class IDs from the cumulative dataset."""
    registry = dataset_root / "classes.json"
    if not registry.exists():
        return []
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
        return [str(x) for x in data.get("names", [])]
    except Exception:
        return []

def save_dataset_yaml(dataset_root, names):
    """Write/update one cumulative Ultralytics dataset.yaml."""
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "classes.json").write_text(
        json.dumps({"names": names}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {json.dumps(name, ensure_ascii=False)}")
    (dataset_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

def generate_dataset_readme(dataset_root, names):
    """Generate cumulative dataset documentation from the actual exported YOLO files."""
    split_stats = {}
    total_images = 0
    total_objects = 0
    negative_images = 0
    class_objects = [0 for _ in names]
    class_images = [set() for _ in names]

    for split in ("train", "val", "test"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        image_count = len([p for p in image_dir.iterdir() if p.is_file()]) if image_dir.exists() else 0
        object_count = 0
        negative_count = 0

        if label_dir.exists():
            for label in label_dir.glob("*.txt"):
                lines = [x.strip() for x in label.read_text(encoding="utf-8").splitlines() if x.strip()]
                if not lines:
                    negative_count += 1
                    continue
                object_count += len(lines)
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue
                    if 0 <= class_id < len(names):
                        class_objects[class_id] += 1
                        class_images[class_id].add(f"{split}/{label.stem}")

        split_stats[split] = {
            "images": image_count,
            "objects": object_count,
            "negative_images": negative_count,
        }
        total_images += image_count
        total_objects += object_count
        negative_images += negative_count

    lines = [
        f"# {dataset_root.name}",
        "",
        "Multi-class object-detection dataset in Ultralytics YOLO format.",
        "",
        "This file is generated automatically by `annotate.py` whenever the dataset is exported or updated.",
        "Do not maintain the class counts manually; they are calculated from the exported YOLO labels.",
        "",
        "## Dataset structure",
        "",
        "```text",
        f"{dataset_root.name}/",
        "├── dataset.yaml",
        "├── classes.json",
        "├── README.md",
        "├── manifests/",
        "├── train/",
        "│   ├── images/",
        "│   └── labels/",
        "├── val/",
        "│   ├── images/",
        "│   └── labels/",
        "└── test/",
        "    ├── images/",
        "    └── labels/",
        "```",
        "",
        "## Classes",
        "",
        "| ID | Class | Images containing class | Object instances |",
        "|---:|---|---:|---:|",
    ]
    for i, name in enumerate(names):
        lines.append(f"| {i} | {name} | {len(class_images[i])} | {class_objects[i]} |")

    lines += [
        "",
        "## Dataset statistics",
        "",
        "| Split | Images | Annotated objects | Negative images |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        s = split_stats[split]
        lines.append(f"| {split} | {s['images']} | {s['objects']} | {s['negative_images']} |")

    lines += [
        f"| **Total** | **{total_images}** | **{total_objects}** | **{negative_images}** |",
        "",
        "## Annotation",
        "",
        "Initial bounding-box proposals are generated with a local Qwen3-VL vision-language model.",
        "Annotations are reviewed in the annotation GUI and can be moved, resized, removed, or added manually before export.",
        "",
        "Images marked as positive contain one or more YOLO bounding boxes.",
        "Images marked as negative are retained as training examples with an empty `.txt` label file.",
        "Rejected images are not included in the exported dataset.",
        "",
        "## YOLO label format",
        "",
        "Each non-empty label line uses:",
        "",
        "```text",
        "<class_id> <x_center> <y_center> <width> <height>",
        "```",
        "",
        "Bounding-box coordinates are normalized to the range 0–1.",
        "",
        "## Class IDs",
        "",
        "`classes.json` preserves the class registry. Existing class IDs must not be reordered after labels have been exported.",
        "`dataset.yaml` is updated automatically when new classes are added.",
        "",
        "## Training",
        "",
        "From the project root, train with:",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe train_yolo_models.py train datasets\<DATASET_NAME>\dataset.yaml --device 0",
        "```",
        "",
        "The training script uses `yolo26n.pt` as the default base model unless another model is supplied with `--model`.",
        "After successful training, the best checkpoint is exported automatically to ONNX unless `--no-onnx` is specified.",
        "",
        "Example with an explicit base model:",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe train_yolo_models.py train datasets\<DATASET_NAME>\dataset.yaml --model <BASE_MODEL>.pt --device 0",
        "```",
        "",
        "## ONNX export",
        "",
        "An existing trained `.pt` model can be exported separately:",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe train_yolo_models.py export models\<RUN_NAME>\weights\best.pt",
        "```",
        "",
        "The resulting `best.onnx` can be used by an ONNX-compatible inference application, subject to matching preprocessing and output decoding.",
        "",
    ]

    (dataset_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def yolo_line(box,cls=0):
    x1,y1,x2,y2=[float(v)/1000 for v in box]
    return f"{cls} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}"

def canvas_size(im,max_width=1050,max_height=720):
    w,h=im.size
    scale=min(max_width/w,max_height/h,1.0)
    return max(1,int(w*scale)),max(1,int(h*scale))

def boxes_to_canvas(boxes,cw,ch):
    objs=[]
    for b in boxes:
        x1,y1,x2,y2=[float(v) for v in b["bbox"]]
        left=x1/1000*cw;top=y1/1000*ch
        width=(x2-x1)/1000*cw;height=(y2-y1)/1000*ch
        objs.append({"type":"rect","version":"4.4.0","originX":"left","originY":"top",
                     "left":left,"top":top,"width":width,"height":height,
                     "fill":"rgba(0,0,0,0)","stroke":"#00ff00","strokeWidth":3,
                     "strokeUniform":True,"scaleX":1,"scaleY":1,"angle":0,
                     "selectable":True,"evented":True,
                     "qwen_source":b.get("source","qwen"),
                     "qwen_confidence":float(b.get("confidence",1.0))})
    return {"version":"4.4.0","objects":objs}

def canvas_to_boxes(data,cw,ch):
    out=[]
    if not data:return out
    for o in data.get("objects",[]):
        # Fabric.js versions differ: older canvas JSON used "rect",
        # Fabric 7 (drawable-canvas >= 0.12) commonly returns "Rect".
        if str(o.get("type","")).lower()!="rect":
            continue
        left=float(o.get("left",0));top=float(o.get("top",0))
        width=float(o.get("width",0))*float(o.get("scaleX",1))
        height=float(o.get("height",0))*float(o.get("scaleY",1))
        # Handle negative scaling and clamp.
        x1=min(left,left+width);x2=max(left,left+width)
        y1=min(top,top+height);y2=max(top,top+height)
        x1=max(0,min(cw,x1));x2=max(0,min(cw,x2))
        y1=max(0,min(ch,y1));y2=max(0,min(ch,y2))
        if x2-x1<2 or y2-y1<2:continue
        source=o.get("qwen_source","manual")
        confidence=float(o.get("qwen_confidence",1.0))
        out.append({"confidence":confidence,"bbox":[x1/cw*1000,y1/ch*1000,x2/cw*1000,y2/ch*1000],"source":source})
    return out

def filtered_boxes(rec,min_conf):
    return [b for b in rec.get("boxes",[]) if b.get("source")=="manual" or float(b.get("confidence",0))>=min_conf]

def set_index(new_idx,total):
    st.session_state.review_index=max(0,min(total-1,int(new_idx)))
    st.session_state.canvas_nonce=st.session_state.get("canvas_nonce",0)+1

def canvas_state_key(rec):
    return "canvas_snapshot_" + str(rec["_json"].resolve())

def persist_snapshot(rec,cw,ch):
    data=st.session_state.get(canvas_state_key(rec))
    if data and isinstance(data,dict):
        boxes=canvas_to_boxes(data,cw,ch)
        if boxes:
            rec["boxes"]=boxes
            save_record(rec)
        return boxes
    return filtered_boxes(rec,0.0)

def commit_canvas(rec,canvas_data,cw,ch):
    boxes=canvas_to_boxes(canvas_data,cw,ch)
    rec["boxes"]=boxes
    rec["decision"]="pending" if rec.get("decision") not in ("negative","rejected") else rec.get("decision")
    save_record(rec)
    return boxes

with st.sidebar:
    base=st.text_input("Ollama URL","http://localhost:11434")
    model=st.text_input("Vision model","huihui_ai/qwen3-vl-abliterated:8b")
    dataset_root=st.text_input("Collector dataset folder","./datasets/BMP-2MD")
    source_subdir=st.selectbox("Source images",["accepted","review","rejected"])
    class_name=st.text_input("YOLO class name","BMP-2MD")
    project_root_input=st.text_input(
        "Project root",
        ".",
        help="Project root. The YOLO dataset is stored under <project root>/datasets/<dataset name>."
    )
    dataset_name=st.text_input(
        "Dataset name",
        "military_object_dataset",
        help="All annotated equipment/classes are accumulated into this one YOLO dataset."
    )
    min_conf=st.slider("Minimum Qwen confidence",0.0,1.0,0.50,0.05)
    max_images=st.number_input("Maximum images this run",1,5000,100,10)
    train_pct=st.slider("Train %",50,90,80,5)
    val_pct=st.slider("Validation %",5,30,10,5)
    st.caption(f"Test % = {100-train_pct-val_pct}%")

root=Path(dataset_root).resolve();source=root/source_subdir
work=root/"annotations_qwen";work.mkdir(parents=True,exist_ok=True)
json_dir=work/"json";json_dir.mkdir(exist_ok=True)

images=[]
if source.exists():
    for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.JPG","*.JPEG","*.PNG","*.WEBP"):images.extend(source.glob(ext))
images=sorted(set(images))
st.write(f"Found **{len(images)}** images in `{source}`.")

force_all = st.checkbox("Force Qwen re-annotation (overwrite cached annotation JSON for this run)", value=False)
if st.button("1. Auto-annotate with Qwen",type="primary",disabled=not images):
    prog=st.progress(0.0);status=st.status("Annotating…",expanded=True)
    limit=min(len(images),int(max_images));done=empty=failures=0
    for i,path in enumerate(images[:limit],1):
        f=json_dir/(path.stem+".json")
        try:
            if force_all or not f.exists():
                boxes=qwen_boxes(base,model,Image.open(path).convert("RGB"),class_name)
                f.write_text(json.dumps({"image":str(path),"class_name":class_name,"boxes":boxes,"decision":"pending"},indent=2),encoding="utf-8")
                if not boxes:empty+=1
            done+=1
        except Exception as e:
            failures+=1;status.write(f"{path.name}: {type(e).__name__}: {e}")
        prog.progress(i/max(1,limit),text=f"{i}/{limit} | empty {empty} | failures {failures}")
    status.update(label=f"Annotation pass complete — {done} processed",state="complete")
    st.rerun()

records=[]
for f in sorted(json_dir.glob("*.json")):
    try:
        r=json.loads(f.read_text(encoding="utf-8"));r["_json"]=f
        if Path(r.get("image","")).exists():
            r.setdefault("boxes",[]);r.setdefault("decision","pending");records.append(r)
    except Exception:pass

if records:
    st.header("2. Review & edit")
    st.caption("Qwen boxes are loaded automatically. Manual drawings are persisted before navigation. Existing annotation JSON files are reused as a disk cache; use Re-run Qwen for this image when you want a fresh automatic annotation.")

    if "review_index" not in st.session_state:st.session_state.review_index=0
    if "canvas_nonce" not in st.session_state:st.session_state.canvas_nonce=0
    st.session_state.review_index=max(0,min(len(records)-1,st.session_state.review_index))
    idx=st.session_state.review_index;rec=records[idx]

    # Navigation is button-driven only; no number_input state fights the buttons.
    n1,n2,n3=st.columns([1,2,1])
    if n1.button("← Previous",disabled=idx==0,use_container_width=True):
        old_im=Image.open(rec["image"]).convert("RGB"); old_cw,old_ch=canvas_size(old_im)
        persist_snapshot(rec,old_cw,old_ch)
        set_index(idx-1,len(records));st.rerun()
    n2.markdown(f"<div style='text-align:center;font-size:1.2rem;padding-top:.4rem'><b>{idx+1} / {len(records)}</b><br>{Path(rec['image']).name}</div>",unsafe_allow_html=True)
    if n3.button("Next →",disabled=idx>=len(records)-1,use_container_width=True):
        old_im=Image.open(rec["image"]).convert("RGB"); old_cw,old_ch=canvas_size(old_im)
        persist_snapshot(rec,old_cw,old_ch)
        set_index(idx+1,len(records));st.rerun()

    im=Image.open(rec["image"]).convert("RGB")
    cw,ch=canvas_size(im)
    bg=im.resize((cw,ch))
    working=filtered_boxes(rec,min_conf)

    left,right=st.columns([4,1.35])
    with right:
        st.markdown("**Bounding-box workflow**")
        st.caption("Green rectangles are Qwen's proposed boxes. Use the canvas toolbar edit/select toggle to move or resize them. If Qwen missed an object, draw an additional rectangle manually.")
        st.write(f"Qwen/current boxes loaded: **{len(working)}**")
        st.write("Decision:",f"**{rec.get('decision','pending')}**")
        st.caption("Tip: select a box in Transform mode and press Delete/Backspace to remove it. If browser/component does not delete it, use Clear all and redraw.")
        if st.button("Re-run Qwen for this image",use_container_width=True):
            try:
                fresh=qwen_boxes(base,model,im,class_name)
                rec["boxes"]=fresh
                rec["decision"]="pending"
                save_record(rec)
                st.session_state.pop(canvas_state_key(rec),None)
                st.session_state.canvas_nonce+=1
                st.rerun()
            except Exception as e:
                st.error(f"Qwen annotation failed: {e}")

        clear=st.button("Clear all boxes",use_container_width=True)
        if clear:
            rec["boxes"]=[];rec["decision"]="pending";save_record(rec)
            st.session_state.pop(canvas_state_key(rec),None)
            st.session_state.canvas_nonce+=1;st.rerun()

    if not working:
        st.warning("Qwen did not provide a usable bounding box for this image at the current confidence threshold. Draw the missing box manually, or mark the image Negative if the target is not present.")

    with left:
        initial=boxes_to_canvas(working,cw,ch)
        # drawable-canvas >= 0.12: Qwen boxes arrive through initial_drawing.
        # "rect" permits manual fallback boxes; toolbar edit/select handles
        # movement/resizing of Qwen or manually drawn rectangles.
        drawing_mode="rect"
        canvas=st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="#00ff00",
            background_image=bg,
            update_streamlit=True,
            height=ch,width=cw,
            drawing_mode=drawing_mode,
            initial_drawing=initial,
            key=f"canvas_{idx}_{st.session_state.canvas_nonce}"
        )
        # drawable-canvas reruns Streamlit after drawing/editing. Keep its latest
        # JSON in session state so a subsequent button click cannot lose it.
        if canvas.json_data is not None:
            # Keep the exact Fabric JSON returned by the component. Parsing is
            # case-insensitive for Fabric 7 object types ("Rect").
            st.session_state[canvas_state_key(rec)]=canvas.json_data

    with right:
        live_data = canvas.json_data or st.session_state.get(canvas_state_key(rec))
        live_boxes = canvas_to_boxes(live_data,cw,ch)
        st.write(f"Canvas boxes detected: **{len(live_boxes)}**")

        if st.button("Save box edits",type="primary",use_container_width=True):
            data=canvas.json_data or st.session_state.get(canvas_state_key(rec))
            commit_canvas(rec,data,cw,ch)
            if data is not None: st.session_state[canvas_state_key(rec)]=data
            st.session_state.canvas_nonce+=1;st.success("Boxes saved.");st.rerun()

        st.divider()
        if st.button("✓ Positive & Next",use_container_width=True):
            data=canvas.json_data or st.session_state.get(canvas_state_key(rec))
            boxes=canvas_to_boxes(data,cw,ch)
            if not boxes:
                st.error("Positive image needs at least one bounding box.")
            else:
                rec["boxes"]=boxes;rec["decision"]="positive";save_record(rec)
                st.session_state.pop(canvas_state_key(rec),None)
                set_index(idx+1,len(records));st.rerun()

        if st.button("Ø Negative & Next",use_container_width=True):
            rec["boxes"]=[];rec["decision"]="negative";save_record(rec)
            st.session_state.pop(canvas_state_key(rec),None)
            set_index(idx+1,len(records));st.rerun()

        if st.button("✗ Reject & Next",use_container_width=True):
            rec["decision"]="rejected";save_record(rec)
            st.session_state.pop(canvas_state_key(rec),None)
            set_index(idx+1,len(records));st.rerun()

        if st.button("Reset decision",use_container_width=True):
            rec["decision"]="pending";save_record(rec);st.rerun()

    positive=[r for r in records if r.get("decision") in ("positive","accepted")]
    negative=[r for r in records if r.get("decision")=="negative"]
    rejected=[r for r in records if r.get("decision")=="rejected"]
    pending=[r for r in records if r.get("decision","pending")=="pending"]
    st.write(f"Positive **{len(positive)}** · Negative **{len(negative)}** · Rejected **{len(rejected)}** · Pending **{len(pending)}**")

    st.header("3. Export")
    project_root=Path(project_root_input).expanduser().resolve()
    dataset_folder=safe_folder_name(dataset_name)
    dataset_root=project_root/"datasets"/dataset_folder
    st.caption("All equipment/classes are accumulated into one YOLO dataset. Existing classes remain in place and dataset.yaml is extended when a new class is exported.")
    st.code(str(dataset_root))
    exportable=positive+negative

    if st.button(f"Export/update YOLO dataset: {dataset_folder}",disabled=not exportable):
        dataset_root.mkdir(parents=True,exist_ok=True)
        for split in ("train","val","test"):
            (dataset_root/split/"images").mkdir(parents=True,exist_ok=True)
            (dataset_root/split/"labels").mkdir(parents=True,exist_ok=True)

        root_classes=load_root_classes(dataset_root)
        if class_name not in root_classes:
            root_classes.append(class_name)
        class_id=root_classes.index(class_name)
        save_dataset_yaml(dataset_root,root_classes)

        # Remove only files previously exported for THIS class/model, so a
        # re-export updates that class without touching any other class.
        prefix=safe_folder_name(class_name)+"__"
        for split in ("train","val","test"):
            for folder in ("images","labels"):
                for old_file in (dataset_root/split/folder).glob(prefix+"*"):
                    if old_file.is_file():
                        old_file.unlink()

        rng=random.Random(42)
        items=list(exportable)
        rng.shuffle(items)
        n=len(items)
        n_train=int(n*train_pct/100)
        n_val=int(n*val_pct/100)
        assignments={}
        for i,rec in enumerate(items):
            split="train" if i<n_train else ("val" if i<n_train+n_val else "test")
            assignments[str(rec["_json"])]=split

        counts={"train":0,"val":0,"test":0}
        positive_count=0
        negative_count=0

        for seq,rec in enumerate(items,1):
            split=assignments[str(rec["_json"])]
            # Annotation records store image paths as strings.
            # Convert to pathlib.Path before using .suffix/copy operations.
            src = Path(rec["image"])
            ext = src.suffix.lower() or ".jpg"
            stem=f"{prefix}{seq:06d}"
            image_dst=dataset_root/split/"images"/f"{stem}{ext}"
            label_dst=dataset_root/split/"labels"/f"{stem}.txt"
            shutil.copy2(src,image_dst)

            if rec.get("decision")=="positive":
                boxes=rec.get("boxes",[])
                label_dst.write_text(
                    "\n".join(yolo_line(b["bbox"],class_id) for b in boxes)+"\n",
                    encoding="utf-8",
                )
                positive_count+=1
            else:
                # Valid YOLO negative image: empty label file.
                label_dst.write_text("",encoding="utf-8")
                negative_count+=1
            counts[split]+=1

        manifest={
            "dataset_name":dataset_folder,
            "dataset_root":str(dataset_root),
            "dataset_yaml":str(dataset_root/"dataset.yaml"),
            "class_name":class_name,
            "class_id":class_id,
            "positive_exported":positive_count,
            "negative_exported":negative_count,
            "rejected":len(rejected),
            "pending":len(pending),
            "splits":counts,
            "classes":root_classes,
        }
        manifests=dataset_root/"manifests"
        manifests.mkdir(exist_ok=True)
        (manifests/f"{safe_folder_name(class_name)}.json").write_text(
            json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8"
        )

        generate_dataset_readme(dataset_root, root_classes)

        st.success(f"Dataset updated: {dataset_root}")
        st.success(f"Dataset README updated: {dataset_root/'README.md'}")
        st.info(f"dataset.yaml: {dataset_root/'dataset.yaml'} | {class_name} = class {class_id}")
        st.json(manifest)

