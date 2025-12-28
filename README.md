# Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion

**Authors:** Hau-Shiang Shiu, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, Yu-Lun Liu

## Abstract
Diffusion-based video super-resolution (VSR) methods achieve strong perceptual quality but remain impractical for latency-sensitive settings due to reliance on future frames and expensive multi-step denoising. We propose Stream-DiffVSR, a causally conditioned diffusion framework for efficient online VSR. Operating strictly on past frames, it combines a four-step distilled denoiser for fast inference, an Auto-regressive Temporal Guidance (ARTG) module injecting motion-aligned cues during latent denoising, and a lightweight temporal-aware decoder with a Temporal Processor Module (TPM) enhancing detail and temporal coherence. Stream-DiffVSR processes 720p frames in 0.328 seconds on an RTX4090 GPU and significantly outperforms prior diffusion-based methods. Compared with the online SOTA TMP~\citep{zhang2024tmp}, it boosts perceptual quality (LPIPS +0.095) while reducing latency by over 130X. Stream-DiffVSR achieves the lowest latency reported for diffusion-based VSR reducing initial delay from over 4600 seconds to 0.328 seconds, thereby making it the first diffusion VSR method suitable for low-latency online deployment.

### 📋 TODO

- ✅ Release inference code and model weights  
- ⬜ Release training code 

## Usage
#### Conda setup
```
git clone https://github.com/jamichss/Stream-DiffVSR.git
cd AR-DiffVSR
conda env create -f requirements.yaml
conda activate stream-diffvsr
```
### Pretrained models
Pretrained models are available [here](https://huggingface.co/Jamichsu/Stream-DiffVSR). You don't need to download them explicitly as they are fetched with inference code.
### Inference
```
bash test_auto_temporal_decoder.sh
```
