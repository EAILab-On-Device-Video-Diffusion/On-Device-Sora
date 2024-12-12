# text to video
# python scripts/inference.py configs/opensora-v1-2/inference/sample.py \
#   --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
#   --prompt "a beautiful waterfall"
python scripts/inference.py configs/opensora-v1-2/inference/sample.py --image-size 128 128 --prompt "a beautiful cow" --num-frames 21
