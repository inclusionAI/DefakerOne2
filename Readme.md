# Venus-DefakerOne2: A Vision‑Forensics Foundation Model with Broader Coverage

### 1. Enviroment Requirements
Python 3.12.3
```
pip install torch==2.9.0 torchvision==0.24.0 ms-swift=4.1.0 numpy==1.26.0 scipy==1.11.3 scikit-learn==1.3.2 torchao==0.16.0
```

### 2. Model Download
```
modelscope download --model DanceText/DefakerOne2 --local_dir [target_save_path]
```
DefakerOne2-preview (in the ``preview'' directory): can be used for reproducing scores of the paper.

DefakerOne2-release (in the ``release'' directory): trained with more data and is expected to perform better in most cases.

### 3. Inference Data Jsonl Line Format
Image
```
{"messages": [{"role": "user", "content": "<image>Is this image real or fake? Answer only Real or Fake"}], "images": ["your_image_path.jpg"]}
```
Video
```
{"messages": [{"role": "user", "content": "<video>Is this video real or fake? Answer only Real or Fake"}], "images": ["your_video_path.mp4"]}
```

### 4. Inference Command
```
CUDA_VISIBLE_DEVICES=0 MIN_PIXELS=262144 MAX_PIXELS=2408448 --model [model_weight_dir] --val_dataset [val_dataset_jsonl_path] --result_path [prediction_jsonl_save_path] --max_new_token 16 --enable_thinking False --batch_size 1 --num_workers 8
```
