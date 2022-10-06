from asyncio.windows_events import NULL
import math
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps, ImageChops

from modules import devices
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts



def process_batch(p, input_dir, output_dir, args):
    processing.fix_seed(p)

    images = [file for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)] if os.path.isfile(file)]

    print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    save_normally = output_dir == ''

    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally

    state.job_count = len(images) * p.n_iter

    for i, image in enumerate(images):
        state.job = f"{i+1} out of {len(images)}"

        if state.interrupted:
            break

        img = Image.open(image)
        p.init_images = [img] * p.batch_size

        proc = modules.scripts.scripts_img2img.run(p, *args)
        if proc is None:
            proc = process_images(p)

        for n, processed_image in enumerate(proc.images):
            filename = os.path.basename(image)

            if n > 0:
                left, right = os.path.splitext(filename)
                filename = f"{left}-{n}{right}"

            if not save_normally:
                processed_image.save(os.path.join(output_dir, filename))


def img2img(init_img, init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, *args, **kwargs):
    mode = kwargs.get('mode', 0)
    is_inpaint = mode == 1
    is_batch = mode == 2

    if is_inpaint:
        if mask_mode == 0:
            image = init_img_with_mask['image']
            mask = init_img_with_mask['mask']
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            image = image.convert('RGB')
        else:
            image = init_img_inpaint
            mask = init_mask_inpaint
    else:
        image = init_img
        mask = None

    assert 0. <= kwargs.get('denoising_strength', 0.7) <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=kwargs.get('prompt',''),
        negative_prompt=kwargs.get('negative_prompt',''),
        styles=[kwargs.get('prompt_style',''), kwargs.get('prompt_style2','')],
        seed=kwargs.get('seed',-1),
        subseed=kwargs.get('subseed',0),
        subseed_strength=kwargs.get('subseed_strength',0.1),
        seed_resize_from_h=kwargs.get('seed_resize_from_h',512),
        seed_resize_from_w=kwargs.get('seed_resize_from_w',512),
        seed_enable_extras=kwargs.get('seed_enable_extras',False),
        sampler_index=kwargs.get('sampler_index',0),
        batch_size=kwargs.get('batch_size',1),
        n_iter=kwargs.get('n_iter',1),
        steps=kwargs.get('steps',20),
        cfg_scale=kwargs.get('cfg_scale',7),
        width=kwargs.get('width',512),
        height=kwargs.get('height',512),
        restore_faces=kwargs.get('restore_faces',False),
        tiling=kwargs.get('tiling',False),
        init_images=[image],
        mask=mask,
        mask_blur=kwargs.get('mask_blur',4),
        inpainting_fill=kwargs.get('inpainting_fill',1),
        resize_mode=kwargs.get('resize_mode',0),
        denoising_strength=kwargs.get('denoising_strength', 0.7),
        inpaint_full_res=kwargs.get('inpaint_full_res',False),
        inpaint_full_res_padding=kwargs.get('inpaint_full_res_padding',32),
        inpainting_mask_invert=kwargs.get('inpainting_mask_invert',False)
    )

    if shared.cmd_opts.enable_console_prompts:
        print(f"\nimg2img: {kwargs.get('prompt','')}", file=shared.progress_print_out)

    p.extra_generation_params["Mask blur"] = kwargs.get('mask_blur',4)

    if is_batch:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

        process_batch(p, kwargs.get('img2img_batch_input_dir',"./"), kwargs.get('img2img_batch_input_dir',"./"), args)

        processed = Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args)
        if processed is None:
            processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)
