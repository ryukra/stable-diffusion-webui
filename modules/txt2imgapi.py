import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


def txt2img(*args,**kwargs):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
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
        enable_hr=kwargs.get('enable_hr',False),
        scale_latent=kwargs.get('scale_latent',False) if kwargs.get('enable_hr',False) else None,
        denoising_strength=kwargs.get('denoising_strength', 0.7) if kwargs.get('enable_hr',False) else None,
    )

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {kwargs.get('prompt','')}", file=shared.progress_print_out)

    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is None:
        processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)

