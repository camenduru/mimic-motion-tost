FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    safetensors einops transformers scipy torchsde aiohttp kornia opencv-python matplotlib scikit-image imageio imageio-ffmpeg ffmpeg-python av fvcore ultralytics \
    omegaconf ftfy accelerate bitsandbytes sentencepiece protobuf diffusers pykalman segment_anything timm insightface addict onnxruntime onnxruntime-gpu yapf && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/kijai/ComfyUI-KJNodes /content/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux /content/ComfyUI/custom_nodes/comfyui_controlnet_aux && \
    git clone https://github.com/cubiq/ComfyUI_essentials /content/ComfyUI/custom_nodes/ComfyUI_essentials && \
    git clone https://github.com/aidenli/ComfyUI_NYJY /content/ComfyUI/custom_nodes/ComfyUI_NYJY && \
    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet /content/ComfyUI/custom_nodes/ComfyUI-Advanced-ControlNet && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus /content/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Layer-norm/comfyui-lama-remover /content/ComfyUI/custom_nodes/comfyui-lama-remover && \
    git clone https://github.com/kijai/ComfyUI-MimicMotionWrapper /content/ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper && \
    git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait /content/ComfyUI/custom_nodes/ComfyUI-AdvancedLivePortrait && \
    git clone https://github.com/storyicon/comfyui_segment_anything /content/ComfyUI/custom_nodes/comfyui_segment_anything && \
    git clone https://github.com/Gourieff/comfyui-reactor-node /content/ComfyUI/custom_nodes/comfyui-reactor-node && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved /content/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved && \
    git clone https://github.com/kijai/ComfyUI-LivePortraitKJ /content/ComfyUI/custom_nodes/ComfyUI-LivePortraitKJ && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation /content/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
    git clone https://github.com/kijai/ComfyUI-LivePortraitKJ /content/ComfyUI/custom_nodes/ComfyUI-LivePortraitKJ && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors -d /content/ComfyUI/models/clip_vision -o CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/MimicMotionMergedUnet_1-1-fp16.safetensors -d /content/ComfyUI/models/mimicmotion -o MimicMotionMergedUnet_1-1-fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors -d /content/ComfyUI/models/controlnet -o control_v11f1p_sd15_depth_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/cyberrealistic_v41BackToBasics.safetensors -d /content/ComfyUI/models/checkpoints -o cyberrealistic_v41BackToBasics.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/cyberrealistic_v60.safetensors -d /content/ComfyUI/models/checkpoints -o cyberrealistic_v60.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/depth_anything_v2_vitl.pth -d /content/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Large -o depth_anything_v2_vitl.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/150_16_swin_l_oneformer_coco_100ep.pth -d /content/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators -o 150_16_swin_l_oneformer_coco_100ep.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/ip-adapter-plus-face_sd15.safetensors -d /content/ComfyUI/models/ipadapter -o ip-adapter-plus-face_sd15.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/test_image.png -d /content/ComfyUI/input -o test_image.png && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/how_far_i_ll_go.mp4 -d /content/ComfyUI/input -o how_far_i_ll_go.mp4 && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/sam_hq_vit_h.pth -d /content/ComfyUI/models/sams -o sam_hq_vit_h.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/groundingdino_swint_ogc.pth -d /content/ComfyUI/models/grounding-dino -o groundingdino_swint_ogc.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/appearance_feature_extractor.safetensors -d /content/ComfyUI/models/liveportrait -o appearance_feature_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/landmark.onnx -d /content/ComfyUI/models/liveportrait -o landmark.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/landmark_model.pth -d /content/ComfyUI/models/liveportrait -o landmark_model.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/motion_extractor.safetensors -d /content/ComfyUI/models/liveportrait -o motion_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/spade_generator.safetensors -d /content/ComfyUI/models/liveportrait -o spade_generator.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/stitching_retargeting_module.safetensors -d /content/ComfyUI/models/liveportrait -o stitching_retargeting_module.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/warping_module.safetensors -d /content/ComfyUI/models/liveportrait -o warping_module.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/1k3d68.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/2d106det.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/det_10g.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o det_10g.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/genderage.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/w600k_r50.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o w600k_r50.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/raw/main/llama/config.json -d /content/ComfyUI/models/llm/Meta-Llama-3.1-8B-bnb-4bit -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/raw/main/llama/generation_config.json -d /content/ComfyUI/models/llm/Meta-Llama-3.1-8B-bnb-4bit -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/llama/model.safetensors -d /content/ComfyUI/models/llm/Meta-Llama-3.1-8B-bnb-4bit -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/raw/main/llama/special_tokens_map.json -d /content/ComfyUI/models/llm/Meta-Llama-3.1-8B-bnb-4bit -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/raw/main/llama/tokenizer.json -d /content/ComfyUI/models/llm/Meta-Llama-3.1-8B-bnb-4bit -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/raw/main/llama/tokenizer_config.json -d /content/ComfyUI/models/llm/Meta-Llama-3.1-8B-bnb-4bit -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/config.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/model.safetensors -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/preprocessor_config.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/special_tokens_map.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/spiece.model -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/tokenizer.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/tokenizer_config.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/raw/main/model_index.json -d /content/ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1 -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/raw/main/vae/config.json -d /content/ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -d /content/ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/vae -o diffusion_pytorch_model.fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/raw/main/scheduler/scheduler_config.json -d /content/ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/raw/main/image_encoder/config.json -d /content/ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/image_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/resolve/main/image_encoder/model.fp16.safetensors -d /content/ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/image_encoder -o model.fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/raw/main/feature_extractor/preprocessor_config.json -d /content/ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/feature_extractor -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/AnimateLCM_sd15_t2v.ckpt -d /content/ComfyUI/models/animatediff_models -o AnimateLCM_sd15_t2v.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MimicMotion/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors -d /content/ComfyUI/models/loras -o AnimateLCM_sd15_t2v_lora.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/jasonot/mycomfyui/resolve/main/rife47.pth -d /content/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife -o rife47.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth -d /content/ComfyUI/models/facerestore_models -o GFPGANv1.3.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth -d /content/ComfyUI/models/facerestore_models -o GFPGANv1.4.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth -d /content/ComfyUI/models/facerestore_models -o codeformer-v0.1.0.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx -d /content/ComfyUI/models/facerestore_models -o GPEN-BFR-512.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx -d /content/ComfyUI/models/facerestore_models -o GPEN-BFR-1024.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-2048.onnx -d /content/ComfyUI/models/facerestore_models -o GPEN-BFR-2048.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.torchscript.pt -d /content/ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper/models/DWPose -o yolox_l.torchscript.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt -d /content/ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper/models/DWPose -o dw-ll_ucoco_384_bs5.torchscript.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -d /content/ComfyUI/models/facedetection -o detection_Resnet50_Final.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth -d /content/ComfyUI/models/facedetection -o parsing_parsenet.pth

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py