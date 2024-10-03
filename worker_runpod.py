import os, json, requests, random, time, runpod

import torch
from PIL import Image
import numpy as np

import nodes
import gc

import asyncio
import execution
import server

from nodes import NODE_CLASS_MAPPINGS
from nodes import load_custom_node
from comfy_extras import nodes_images, nodes_mask, nodes_differential_diffusion, nodes_post_processing
import model_management

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server)

load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-KJNodes")
load_custom_node("/content/ComfyUI/custom_nodes/comfyui_controlnet_aux")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_essentials")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_NYJY")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Advanced-ControlNet")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus")
load_custom_node("/content/ComfyUI/custom_nodes/comfyui-lama-remover")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper")
load_custom_node("/content/ComfyUI/custom_nodes/comfyui_segment_anything")
load_custom_node("/content/ComfyUI/custom_nodes/comfyui-reactor-node")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-LivePortraitKJ")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation")

IPAdapterModelLoader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
IPAdapterAdvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
VHS_LoadVideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
VHS_VideoCombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
ImageResizeKJ = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
VHS_SelectEveryNthImage = NODE_CLASS_MAPPINGS["VHS_SelectEveryNthImage"]()
VHS_SelectImages = NODE_CLASS_MAPPINGS["VHS_SelectImages"]()
DepthAnythingV2Preprocessor = NODE_CLASS_MAPPINGS["DepthAnythingV2Preprocessor"]()
OneFormer_COCO_SemSegPreprocessor = NODE_CLASS_MAPPINGS["OneFormer-COCO-SemSegPreprocessor"]()
MaskFromColor = NODE_CLASS_MAPPINGS["MaskFromColor+"]()
JoyCaption = NODE_CLASS_MAPPINGS["JoyCaption"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
ControlNetLoader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
ACN_AdvancedControlNetApply = NODE_CLASS_MAPPINGS["ACN_AdvancedControlNetApply"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
LamaRemover = NODE_CLASS_MAPPINGS["LamaRemover"]()
MimicMotionSampler = NODE_CLASS_MAPPINGS["MimicMotionSampler"]()
DownloadAndLoadMimicMotionModel = NODE_CLASS_MAPPINGS["DownloadAndLoadMimicMotionModel"]()
MimicMotionGetPoses = NODE_CLASS_MAPPINGS["MimicMotionGetPoses"]()
DiffusersScheduler = NODE_CLASS_MAPPINGS["DiffusersScheduler"]()
MimicMotionDecode = NODE_CLASS_MAPPINGS["MimicMotionDecode"]()
SAMModelLoader = NODE_CLASS_MAPPINGS["SAMModelLoader (segment anything)"]()
GroundingDinoModelLoader = NODE_CLASS_MAPPINGS["GroundingDinoModelLoader (segment anything)"]()
GroundingDinoSAMSegment = NODE_CLASS_MAPPINGS["GroundingDinoSAMSegment (segment anything)"]()
RepeatImageBatch = nodes_images.NODE_CLASS_MAPPINGS["RepeatImageBatch"]()
MaskBlur = NODE_CLASS_MAPPINGS["MaskBlur+"]()
ImageComposite = NODE_CLASS_MAPPINGS["ImageComposite+"]()
MaskToImage = nodes_mask.NODE_CLASS_MAPPINGS["MaskToImage"]()
ReActorFaceSwap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
ApplyAnimateDiffModelBasicNode = NODE_CLASS_MAPPINGS["ADE_ApplyAnimateDiffModelSimple"]()
LoadAnimateDiffModelNode = NODE_CLASS_MAPPINGS["ADE_LoadAnimateDiffModel"]()
LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
StandardStaticContextOptionsNode = NODE_CLASS_MAPPINGS["ADE_StandardStaticContextOptions"]()
SampleSettingsNode = NODE_CLASS_MAPPINGS["ADE_AnimateDiffSamplingSettings"]()
UseEvolvedSamplingNode = NODE_CLASS_MAPPINGS["ADE_UseEvolvedSampling"]()
DifferentialDiffusion = nodes_differential_diffusion.NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
ConditioningZeroOut = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
SetLatentNoiseMask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
InvertMask = nodes_mask.NODE_CLASS_MAPPINGS["InvertMask"]()
RemapMaskRange = NODE_CLASS_MAPPINGS["RemapMaskRange"]()
RIFE_VFI =  NODE_CLASS_MAPPINGS["RIFE VFI"]()
DownloadAndLoadLivePortraitModels = NODE_CLASS_MAPPINGS["DownloadAndLoadLivePortraitModels"]()
LivePortraitLoadCropper = NODE_CLASS_MAPPINGS["LivePortraitLoadCropper"]()
LivePortraitCropper = NODE_CLASS_MAPPINGS["LivePortraitCropper"]()
LivePortraitProcess = NODE_CLASS_MAPPINGS["LivePortraitProcess"]()
LivePortraitComposite = NODE_CLASS_MAPPINGS["LivePortraitComposite"]()
LivePortraitRetargeting = NODE_CLASS_MAPPINGS["LivePortraitRetargeting"]()
CreateShapeMask = NODE_CLASS_MAPPINGS["CreateShapeMask"]()
GrowMaskWithBlur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
ImageSharpen = nodes_post_processing.NODE_CLASS_MAPPINGS["ImageSharpen"]()

with torch.inference_mode():
    ipadapter = IPAdapterModelLoader.load_ipadapter_model("ip-adapter-plus-face_sd15.safetensors")[0]
    clip_vision = CLIPVisionLoader.load_clip("CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")[0]
    unet, clip, vae = CheckpointLoaderSimple.load_checkpoint("cyberrealistic_v60.safetensors")
    control_net = ControlNetLoader.load_controlnet(control_net_name="control_v11f1p_sd15_depth_fp16.safetensors")[0]
    mimic_pipeline = DownloadAndLoadMimicMotionModel.loadmodel('fp16', 'MimicMotionMergedUnet_1-1-fp16.safetensors')[0]
    sam_model = SAMModelLoader.main("sam_hq_vit_h (2.57GB)")[0]
    grounding_dino_model = GroundingDinoModelLoader.main("GroundingDINO_SwinT_OGC (694MB)")[0]
    animate_diff_lora = LoraLoaderModelOnly.load_lora_model_only(unet, "AnimateLCM_sd15_t2v_lora.safetensors", 1.0)[0]
    motion_model = LoadAnimateDiffModelNode.load_motion_model(model_name="AnimateLCM_sd15_t2v.ckpt", ad_settings=None)[0]
    pipeline = DownloadAndLoadLivePortraitModels.loadmodel(precision="fp16", mode="human")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='test_image')
    seed = values['seed']
    max_new_tokens = values['max_new_tokens']
    controlnet_strength = values['controlnet_strength']
    context_size = values['context_size']
    video_file = values['video_file']
    
    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    video_images, frame_count, audio, video_info = VHS_LoadVideo.load_video(video=video_file, force_rate=30, force_size="Disabled", custom_width=512, custom_height=512, frame_load_cap=142, skip_first_frames=1350, select_every_nth=1)
    source_image = ImageResizeKJ.resize(video_images, width=512, height=768, keep_proportion=False, upscale_method="nearest-exact", divisible_by=2, crop="center")[0]
    source_image = VHS_SelectEveryNthImage.select_images(images=source_image, select_every_nth=2, skip_first_images=0)[0]
    source_image_count = source_image.size(0)
    depth = DepthAnythingV2Preprocessor.execute(source_image, "depth_anything_v2_vitl.pth", resolution=512)[0]
    coco = OneFormer_COCO_SemSegPreprocessor.semantic_segmentate(source_image, resolution=512)[0]
    mask_from_color = MaskFromColor.execute(coco, 220, 20, 60, 5)[0]
    src_image = os.path.basename(input_image)
    prompt = "A descriptive caption for this image"
    top_k = 10
    temperature = 0.50
    clear_cache = True
    src_image = LoadImage.load_image(src_image)[0]
    positive_prompt = JoyCaption.run_local("Meta-Llama-3.1-8B-bnb-4bit", src_image, prompt, max_new_tokens, top_k, temperature, clear_cache)[0]
    negative_prompt = "text, watermark"
    cond = nodes.CLIPTextEncode().encode(clip, positive_prompt)[0]
    n_cond = nodes.CLIPTextEncode().encode(clip, negative_prompt)[0]
    start_percent = 0
    end_percent = 1
    positive, negative, model_optional = ACN_AdvancedControlNetApply.apply_controlnet(positive=cond, negative=n_cond, control_net=control_net, image=depth, mask_optional=mask_from_color, strength=controlnet_strength, start_percent=start_percent, end_percent=end_percent)
    ipadapter_model = IPAdapterAdvanced.apply_ipadapter(unet, ipadapter, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", 
                    combine_embeds="concat", weight_faceidv2=None, image=src_image, image_style=None, image_composition=None, image_negative=None, clip_vision=clip_vision,
                    attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None,
                    composition_boost=None, enhance_tiles=1, enhance_ratio=1.0, weight_kolors=1.0)[0]
    height = 768
    width = 512
    latent = {"samples":torch.zeros([1, 4, height // 8, width // 8])}
    steps = 50
    cfg = 7.5
    sampler_name = "dpmpp_2m"
    scheduler = "karras"
    sample = nodes.common_ksampler(model=ipadapter_model,
                                    seed=seed,
                                    steps=steps,
                                    cfg=cfg,
                                    sampler_name=sampler_name,
                                    scheduler=scheduler,
                                    positive=positive,
                                    negative=negative,
                                    latent=latent,
                                    denoise=1.0)

    
    sample_in = sample[0]["samples"].to(torch.float16).cuda()
    ref_image = vae.decode(sample_in)
    ref_image = ref_image.cpu()
    mask_threshold = 250
    gaussblur_radius = 20
    invert_mask = False
    lamda_remover_images = LamaRemover.lama_remover(ref_image, mask_from_color, mask_threshold, gaussblur_radius, invert_mask)[0]
    pose_with_ref, pose_images = MimicMotionGetPoses.process(ref_image, source_image, True, True, True)
    scheduler = 'EulerDiscreteScheduler'
    sigma_min = 0.002
    sigma_max = 700
    align_your_steps = True
    optional_scheduler = DiffusersScheduler.loadmodel(scheduler, sigma_min, sigma_max, align_your_steps)[0]
    cfg_min = 2.0
    cfg_max = 2.0
    steps = 10
    noise_aug_strength = 0
    fps = 15
    keep_model_loaded = False
    context_overlap = 6
    mimic_motion_samples = MimicMotionSampler.process(mimic_pipeline, ref_image, pose_with_ref, cfg_min, cfg_max, steps, seed, noise_aug_strength, fps, keep_model_loaded, 
                                                        context_size, context_overlap, optional_scheduler=optional_scheduler, pose_strength=1.0, image_embed_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0)[0]
    decode_chunk_size = 4
    mimic_motion_samples_decoded = MimicMotionDecode.process(mimic_pipeline, mimic_motion_samples, decode_chunk_size)[0]
    image_masks, sam_masks = GroundingDinoSAMSegment.main(grounding_dino_model, sam_model, mimic_motion_samples_decoded, "person", 0.30)
    repeat_image_batch = RepeatImageBatch.repeat(lamda_remover_images, source_image_count)[0]
    mask_blur = MaskBlur.execute(sam_masks, 6, "auto")
    image_composite = ImageComposite.execute(repeat_image_batch, mimic_motion_samples_decoded, 0, 0, 0, 0, mask=mask_blur[0])[0]
    enabled = True
    swap_model = "inswapper_128.onnx"
    detect_gender_source = False
    detect_gender_input = False
    source_faces_index = "0"
    input_faces_index = "0"
    console_log_level = 1
    face_restore_model = "GFPGANv1.4.pth"
    face_restore_visibility = 1
    codeformer_weight = 0.50
    facedetection = "retinaface_resnet50"
    mask_from_color_to_image = MaskToImage.mask_to_image(mask_from_color)[0]
    face_swap_image = ReActorFaceSwap.execute(enabled=True, input_image=image_composite, swap_model=swap_model, detect_gender_source=detect_gender_source, detect_gender_input=detect_gender_input, source_faces_index=source_faces_index, 
                                                input_faces_index=input_faces_index, console_log_level=console_log_level, face_restore_model=face_restore_model, face_restore_visibility=face_restore_visibility, codeformer_weight=codeformer_weight, 
                                                facedetection=facedetection, source_image=mask_from_color_to_image, face_model=None, faces_order=None, face_boost=None)[0]
    animate_diff = ApplyAnimateDiffModelBasicNode.apply_motion_model(motion_model=motion_model, motion_lora=None, scale_multival=1.0, effect_multival=None, ad_keyframes=None, per_block=None)[0]
    context_options = StandardStaticContextOptionsNode.create_options(context_length=16, context_overlap=6, fuse_method="pyramid", use_on_equal_length=False, start_percent=0.0, guarantee_steps=1, view_opts=None, prev_context=None)[0]
    sample_settings = SampleSettingsNode.create_settings(batch_offset=0, noise_type="FreeNoise", seed_gen="comfy", seed_offset=0, noise_layers=None, iteration_opts=None, seed_override=None, adapt_denoise_steps=False, custom_cfg=None, 
                                                        sigma_schedule=None, image_inject=None)[0]
    evolved_model = UseEvolvedSamplingNode.use_evolved_sampling(model=animate_diff_lora, beta_schedule="lcm avg(sqrt_linear,linear)", m_models=animate_diff, context_options=context_options, sample_settings=sample_settings, beta_schedule_override=None)[0]
    differential_diffusion_model = DifferentialDiffusion.apply(evolved_model)[0]
    positive = nodes.CLIPTextEncode().encode(clip, "")[0]
    negative = ConditioningZeroOut.zero_out(cond)[0]
    vae_encode = VAEEncode.encode(vae, face_swap_image)[0]
    invert_mask = InvertMask.invert(mask_blur[0])[0]
    remaped_mask = RemapMaskRange.remap(invert_mask, 0.30, 0.50)[0]
    latent_noise_mask = SetLatentNoiseMask.set_mask(vae_encode, remaped_mask)[0]
    steps = 4
    cfg = 1.0
    sampler_name = "lcm"
    scheduler = "sgm_uniform"
    sample = nodes.common_ksampler(model=differential_diffusion_model,
                                    seed=seed,
                                    steps=steps,
                                    cfg=cfg,
                                    sampler_name=sampler_name,
                                    scheduler=scheduler,
                                    positive=positive,
                                    negative=negative,
                                    latent=latent_noise_mask,
                                    denoise=1.0)
    sample_in = sample[0]["samples"].to(torch.float16).cuda()
    diff_image = vae.decode(sample_in)
    diff_image = diff_image.cpu()
    rife_vfi = RIFE_VFI.vfi(frames=diff_image, ckpt_name="rife47.pth", clear_cache_after_n_frames=10, multiplier=2, fast_mode=True, ensemble=True, scale_factor = 1.0)[0]
    cropper = LivePortraitLoadCropper.crop("CUDA", True, detection_threshold=0.5)[0]
    cropped_image1, crop_info1 = LivePortraitCropper.process(pipeline, cropper, rife_vfi, 512, 2.30, 0.0, -0.125, 0, "large-small", True)
    cropped_image2, crop_info2 = LivePortraitCropper.process(pipeline, cropper, video_images, 512, 2.30, 0.0, -0.125, 0, "large-small", True)
    opt_retargeting_info = LivePortraitRetargeting.process(driving_crop_info=crop_info2, eye_retargeting=False, eyes_retargeting_multiplier=1.0, lip_retargeting=True, lip_retargeting_multiplier=1.0)[0]
    lip_zero=False
    lip_zero_threshold=0.03
    stitching=True
    relative_motion_mode="source_video_smoothed"
    driving_smooth_observation_variance=0.000003
    delta_multiplier=1.0
    mismatch_method="constant"
    expression_friendly=False
    expression_friendly_multiplier=1.0
    cropped_image, liveportrait_out = LivePortraitProcess.process(source_image=rife_vfi,
                                                                    driving_images=cropped_image2,
                                                                    crop_info=crop_info1,
                                                                    pipeline=pipeline,
                                                                    lip_zero=lip_zero,
                                                                    lip_zero_threshold=lip_zero_threshold,
                                                                    stitching=stitching,
                                                                    relative_motion_mode=relative_motion_mode,
                                                                    driving_smooth_observation_variance=driving_smooth_observation_variance,
                                                                    delta_multiplier=delta_multiplier,
                                                                    mismatch_method=mismatch_method,
                                                                    opt_retargeting_info=opt_retargeting_info,
                                                                    expression_friendly=expression_friendly,
                                                                    expression_friendly_multiplier=expression_friendly_multiplier)
    create_shape_mask = CreateShapeMask.createshapemask(frames=1, frame_width=512, frame_height=512, location_x=256, location_y=256, shape_width=472, shape_height=472, grow=0, shape="circle")[0]
    grow_mask_with_blur = GrowMaskWithBlur.expand_mask(mask=create_shape_mask, expand=0, tapered_corners=True, flip_input=False, blur_radius=13.9, incremental_expandrate=0.0, lerp_alpha=1.00, decay_factor=1.0, fill_holes=False)[0]
    full_images = LivePortraitComposite.process(source_image=rife_vfi, cropped_image=cropped_image, liveportrait_out=liveportrait_out, mask=grow_mask_with_blur)[0]
    image_sharpen = ImageSharpen.sharpen(image=full_images, sharpen_radius=1, sigma=1.0, alpha=0.20)[0]
    combined_video = VHS_VideoCombine.combine_video(images=image_sharpen, audio=audio, frame_rate=30, loop_count=0, filename_prefix="MimicMotion", format="video/h264-mp4", save_output=True, prompt=None, unique_id=None)

    source = combined_video["result"][0][1][1]
    source_with_audio = source.replace(".mp4", "-audio.mp4")
    source_with_png = source.replace(".mp4", ".png")
    destination = '/content/ComfyUI/output/mimic-motion-tost.mp4'
    destination_with_audio = '/content/ComfyUI/output/mimic-motion-audio-tost.mp4'
    destination_with_png = '/content/ComfyUI/output/mimic-motion-tost.png'
    shutil.move(source, destination)
    shutil.move(source_with_audio, destination_with_audio)
    shutil.move(source_with_png, destination_with_png)
    
    result = destination_with_audio
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)

runpod.serverless.start({"handler": generate})