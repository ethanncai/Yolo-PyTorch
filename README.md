# Yolo-Pytorch

This is a simple project that implement yolov11 family using just Torch.

* [Done] Yolo Model rewrite
* [Done] Yolo Inference rewrite
* [Done] Yolo Trainer

Train with one dataset:

```bash
python train.py --data dataset.yaml --epochs 100 --batch 16
```

By default, the training pipeline keeps only `face` and `person`, remaps them to
`face=0, person=1`, and ignores `hand` labels. This is the recommended simpler
association setup:

```bash
python train.py --data dataset.yaml --weights assets/ckpts/yolo11n.ckpt --epochs 100 --batch 16 --keep-names face,person
python infer.py assets/ckpts/face_person_pair_v1_best.ckpt image.jpg -o out.jpg
```

Video inference uses ByteTrack from `/home/junzhicai/ByteTrack` for person tracks;
faces inherit the matched person track/color. Run it in the `vlm_sft` conda env:

```bash
conda activate vlm_sft
python infer_video.py assets/ckpts/face_person_pair_v1_best_det.ckpt input.mp4 -o tracked.mp4 \
	--device cuda --batch-size 4 --conf 0.25 \
	--track-low-conf 0.1 --track-high-conf 0.5 --new-track-conf 0.6
```

For video speed, keep `--batch-size 4` on a V100-like GPU. `--half` is available,
but was not faster in the current CUDA/PyTorch setup; the remaining bottleneck is
mostly video IO, drawing, NMS, and tracker/post-processing rather than raw model
matrix compute.

Pass `--keep-names ""` to train or infer with all checkpoint/data classes.

The association branch is trained as a face-person pair scorer: after one YOLO
forward pass, the model scores every detected face/person pair from their box
embeddings plus relative geometry. Training uses BCE supervision from
`person_id`; inference matches faces to people from the pair score matrix. Old
checkpoints without `pair_scorer` weights still run, but fall back to geometry-only
association.

Train with multiple datasets. The `names` and class order in each `data.yaml` must be the same:

```bash
python train.py --data dataset1.yaml dataset2.yaml --epochs 100 --batch 16
```
