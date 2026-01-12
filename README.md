# Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion

**Authors:** Hau-Shiang Shiu, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, Yu-Lun Liu

<a href='https://jamichss.github.io/stream-diffvsr-project-page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/Jamichsu/Stream-DiffVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1)-blue"></a> &nbsp;
<a href="https://arxiv.org/abs/2512.23709"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>

### TODO

- ✅ Release inference code and model weights  
- ⬜ Release training code 

## Abstract
Diffusion-based video super-resolution (VSR) methods achieve strong perceptual quality but remain impractical for latency-sensitive settings due to reliance on future frames and expensive multi-step denoising. We propose Stream-DiffVSR, a causally conditioned diffusion framework for efficient online VSR. Operating strictly on past frames, it combines a four-step distilled denoiser for fast inference, an Auto-regressive Temporal Guidance (ARTG) module injecting motion-aligned cues during latent denoising, and a lightweight temporal-aware decoder with a Temporal Processor Module (TPM) enhancing detail and temporal coherence. Stream-DiffVSR processes 720p frames in 0.328 seconds on an RTX4090 GPU and significantly outperforms prior diffusion-based methods. Compared with the online SOTA TMP~\citep{zhang2024tmp}, it boosts perceptual quality (LPIPS +0.095) while reducing latency by over 130X. Stream-DiffVSR achieves the lowest latency reported for diffusion-based VSR reducing initial delay from over 4600 seconds to 0.328 seconds, thereby making it the first diffusion VSR method suitable for low-latency online deployment.

## Usage

### Environment 
The code is based on Python 3.9, CUDA 11, and [diffusers](https://github.com/huggingface/diffusers), and our development and testing are primarily conducted on Ubuntu 24.04 LTS.

### Conda setup (base)
```
git clone https://github.com/jamichss/Stream-DiffVSR.git
cd Stream-DiffVSR
conda env create -f requirements.yml
conda activate stream-diffvsr
```

### CLI inference (frames)
Use the base conda environment above and run `inference.py` as shown in the next section.

### Gradio UI (local)
Use the base conda environment above, then install the UI requirements:
```
pip install -r requirements-app.txt
```
This installs the Gradio UI dependencies and pins a compatible `huggingface-hub` for the UI path. It is not required for CLI inference.
### Pretrained models
Pretrained models are available [here](https://huggingface.co/Jamichsu/Stream-DiffVSR). You don't need to download them explicitly as they are fetched with inference code.
### Inference
You can run the inference directly using the following command. No manual download of checkpoints is required, as the inference script will automatically fetch the necessary files.
```
python inference.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path 'YOUR_OUTPUT_PATH' \
    --in_path 'YOUR_INPUT_PATH' \
    --num_inference_steps 4
```
The expected file structure for the inference input data is outlined below. The model processes individual video sequences contained within subdirectories.
```
YOUR_INPUT_PATH/
├── seq1/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── seq2/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
```
For additional acceleration using NVIDIA TensorRT, please execute the following command. Please note that utilizing TensorRT may introduce a slight degradation in the output quality while providing significant performance gains. Parameters image_height and image_width are required when using tensorRT; otherwise, they are not needed.

**Note:** **TensorRT** is mainly for speed/throughput, while **xFormers** helps reduce GPU memory usage. They are currently not compatible, so xFormers-based memory optimizations are unavailable when TensorRT is enabled, which may significantly increase GPU memory usage and lead to OOM issues at higher resolutions.

```
python inference.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path 'YOUR_OUTPUT_PATH' \
    --in_path 'YOUR_INPUT_PATH' \
    --num_inference_steps 4 \
    --enable_tensorrt \
    --image_height <YOUR_OUTPUT_HEIGHT> \
    --image_width <YOUR_OUTPUT_WIDTH>
```

When executing the TensorRT command for the first time with a new output resolution, you may observe that the process takes an extended period to build the dedicated TensorRT engine. We kindly ask for your patience. Please note that this engine compilation is a one-time setup step for that specific resolution, essential for enabling subsequent accelerated inference at the same setting.

## Gradio UI (Video Upscaling)

The project includes a simple Gradio app which accepts a source video, extracts frames, runs Stream-DiffVSR on the frames, rebuilds a video at the original FPS, and muxes the original audio back in. A preview and download button are provided for the output.

### Requirements
- CUDA-capable GPU recommended
- `ffmpeg` available on the host

### Run locally
```
python app.py
```
Then open `http://localhost:8000` in your browser.

### Install paths summary
- CLI inference: `requirements.yml` (conda base).
- Gradio UI: `requirements.yml` + `requirements-app.txt`.
- Docker: `Dockerfile` + `requirements-app.txt` (installs CUDA PyTorch wheels and UI deps).

### Environment variables
- `STREAM_DIFFVSR_MODEL_ID`: override the Hugging Face model ID (default `Jamichsu/Stream-DiffVSR`)
- `STREAM_DIFFVSR_TMP_DIR`: temp workspace for extracted frames (default `.gradio_tmp`)
- `STREAM_DIFFVSR_OUTPUT_DIR`: output folder for final videos (default `outputs`)
- `PORT`: server port (default `8000`)

## Docker

Build the image:
```
docker build -t stream-diffvsr-gradio .
```

Run with NVIDIA GPU access:
```
docker run --gpus all -p 8000:8000 stream-diffvsr-gradio
```

Then open `http://localhost:8000` in your browser.

## Diagnostics

If you need to report a dependency issue, you can capture a full environment snapshot (optional):
```
pip list --format=freeze > requirements-full.txt
```
This file is not used for installation, only diagnostics.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{shiu2025stream,
  title={Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion},
  author={Shiu, Hau-Shiang and Lin, Chin-Yang and Wang, Zhixiang and Hsiao, Chi-Wei and Yu, Po-Fan and Chen, Yu-Chih and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2512.23709},
  year={2025}
}
```

<!--## Acknowledgement
This project is built upon the following open-source projects: [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [StableVSR](https://github.com/claudiom4sir/StableVSR) and [TAESD](https://github.com/madebyollin/taesd). We thank all the authors for their great repos.-->
