# Yolo-Pytorch Cheatsheet

Pure PyTorch YOLO11 detection + face/hand/person association.

## Env

```bash
cd /home/junzhicai/Yolo-Pytorch
source /home/junzhicai/anaconda3/etc/profile.d/conda.sh
conda activate vlm_sft
```

## 1. Train Without Hand

Keeps only `face,person`. Labels must be normal YOLO labels or association labels with `person_id` as column 6.

```bash
python train.py \
  --data /path/to/assoc_face_person/data.yaml \
  --weights assets/ckpts/yolo11n.ckpt \
  --scale n \
  --imgsz 640 \
  --batch 16 \
  --epochs 100 \
  --device cuda \
  --project runs/train \
  --name face_person_no_hand \
  --keep-names face,person \
  --assoc 0.1 \
  --val-interval 1 \
  --patience 50
```

## 2. Train With Hand

Keeps `hand,face,person`. The association loss trains both `face -> person` and `hand -> person` from `person_id`.

```bash
python train.py \
  --data /path/to/assoc_hand_face_person/data.yaml \
  --weights assets/ckpts/yolo11n.ckpt \
  --scale n \
  --imgsz 640 \
  --batch 16 \
  --epochs 100 \
  --device cuda \
  --project runs/train \
  --name hand_face_person \
  --keep-names hand,face,person \
  --assoc 0.1 \
  --val-interval 1 \
  --patience 50
```

## 3. Image Inference With Hand

```bash
python infer.py \
  runs/train/hand_face_person/weights/best_det.ckpt \
  /path/to/image.jpg \
  -o outputs/infer_hand_image.jpg \
  --device cuda \
  --imgsz 640 \
  --conf 0.25 \
  --face-conf 0.56 \
  --iou 0.45 \
  --assoc-thres 0.45 \
  --keep-names hand,face,person
```

## 4. Video Inference With Hand

Person is tracked by ByteTrack. Face and hand passively inherit the matched person track id/color.

```bash
python infer_video.py \
  runs/train/hand_face_person/weights/best_det.ckpt \
  /path/to/input.mp4 \
  -o outputs/infer_hand_video.mp4 \
  --device cuda \
  --batch-size 4 \
  --imgsz 640 \
  --conf 0.25 \
  --face-conf 0.56 \
  --iou 0.45 \
  --assoc-thres 0.45 \
  --keep-names hand,face,person \
  --track-low-conf 0.1 \
  --track-high-conf 0.5 \
  --new-track-conf 0.6 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --track-assign-iou 0.3
```

## 5. Image Inference Without Hand

```bash
python infer.py \
  assets/ckpts/face_person_pair_v1_best_det.ckpt \
  /path/to/image.jpg \
  -o outputs/infer_no_hand_image.jpg \
  --device cuda \
  --imgsz 640 \
  --conf 0.25 \
  --face-conf 0.56 \
  --iou 0.45 \
  --assoc-thres 0.45 \
  --keep-names face,person
```

## 6. Video Inference Without Hand

```bash
python infer_video.py \
  assets/ckpts/face_person_pair_v1_best_det.ckpt \
  /path/to/input.mp4 \
  -o outputs/infer_no_hand_video.mp4 \
  --device cuda \
  --batch-size 4 \
  --imgsz 640 \
  --conf 0.25 \
  --face-conf 0.56 \
  --iou 0.45 \
  --assoc-thres 0.45 \
  --keep-names face,person \
  --track-low-conf 0.1 \
  --track-high-conf 0.5 \
  --new-track-conf 0.6 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --track-assign-iou 0.3
```

## 7. Convert YOLO Dataset To Association Dataset

Input: normal YOLO `hand,face,person` dataset with label rows `cls x y w h`.
Output: association dataset with label rows `cls x y w h person_id`.
The output `person` boxes are replaced by YOLO11 pose person boxes; source
dataset `person` boxes are ignored because they are often noisy.

```bash
python scripts/gen_association_dataset.py \
  --src /path/to/normal_hand_face_person/data.yaml \
  --out outputs/assoc_hand_face_person \
  --pose-weights assets/ckpts/yolo11m-pose.pt \
  --device 0 \
  --pose-imgsz 640 \
  --pose-conf 0.25 \
  --pose-kpt-conf 0.25 \
  --pose-min-score 0.0 \
  --person-names person,body \
  --face-names face,head \
  --hand-names hand,left_hand,right_hand \
  --face-hand-max-dist 0.45 \
  --fraction 1.0 \
  --viz 32 \
  --copy-mode auto \
  --overwrite
```

After conversion, inspect:

```bash
ls outputs/assoc_hand_face_person/viz
```
