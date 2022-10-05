import os
import threading

from modules.paths import script_path

import signal

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.ui
import modules.scripts
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.ldsr_model as ldsr
import modules.extras
import modules.lowvram
from modules.txt2img import *
from modules.img2img import *
import modules.sd_models
import modules.swinir as swinir
import time
from io import BytesIO
from PIL import Image
import json
import base64
from typing import Union
from pydantic import BaseModel
import gdiffusion
from modules.sd_samplers import samplers, samplers_for_img2img,samplers_k_diffusion


class apiImage(BaseModel):
    # these are base 64 encoded image data
    image: str = ""
    mask: str = ""
    def __getitem__(self, item):
        return getattr(self, item)

class apiInput(BaseModel):
    prompt: str = 'error sign on a green field with big clouds in the back, by greg rutkowski'
    neg_prompt: str = ''
    mode: str = 'Not used anymore'
    steps: int = 30
    sampler: str = 'LMS'
    mask_blur: int = 4
    inpainting_fill: int =2 
    use_gfpgan: bool = False
    batch_count: int = 1
    cfg_scale: float = 7.0
    denoising_strength: float = 1.0
    seed: int = -1
    height: int = 512
    width: int = 512
    resize_mode: int = 0 # not sure what this one is
    upscaler: str = ''
    upscale_overlap: int = 64
    inpaint_full_res: bool = True
    inpainting_mask_invert: int = 0 # should be bool
    restore_faces: bool = False
    tiling: bool = False

    def __getitem__(self, item):
        return getattr(self, item)

class apiInputPlusImage(apiInput):
    initimage: apiImage = None

modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
swinir.load_models(cmd_opts.swinir_models_path)
realesrgan.setup_realesrgan()
ldsr.add_lsdr()

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):
        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        return res

    return modules.ui.wrap_gradio_call(f)


modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

shared.sd_model = modules.sd_models.load_model()
shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))


def api():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    demo = modules.ui.create_ui(
        txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
        img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
        run_extras=wrap_gradio_gpu_call(modules.extras.run_extras),
        run_pnginfo=modules.extras.run_pnginfo
    )

    (app, local, gradio_remote) = demo.launch(
        share=cmd_opts.share,
        server_name="0.0.0.0" if cmd_opts.listen else None,
        server_port=cmd_opts.port,
        debug=cmd_opts.gradio_debug,
        auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
        inbrowser=cmd_opts.autolaunch,
        prevent_thread_lock=True
    )
    @app.get("/api/version")
    def processVersion():
        print("test")
        return {'version':2.1} 

    @app.get("/api/test")
    def processTest():
        return {"message": "Hello World"}

    @app.post("/api/")
    async def processAPI(data: apiInputPlusImage):
        oimages = []
        oinfo = []
        smp_index=0
        for i in range(0,len(samplers_k_diffusion)):
            if samplers_k_diffusion[i][0]==data['sampler']: smp_index=i    
        
        if data['mode'] == 'txt2img':

#def txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, 
#restore_faces: bool, 
#tiling: bool, 
#n_iter: int, batch_size: int, cfg_scale: float, seed: int,
# subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, 
# seed_enable_extras: bool, height: int, width: int, enable_hr: bool, scale_latent: bool, denoising_strength: float, *args):


            # only positional arguments allowed because of *args 
            oimages, oinfo, ohtml = txt2img(
                data['prompt'],data['neg_prompt'], "","",data['steps'], smp_index,
                data['restore_faces'], 
                data['tiling'],
                data['batch_count'],1,data['cfg_scale'],data['seed'],
                0,0.0, 0, 0,False,
                data['height'], 
                data['width'],
                False,
                False,
                0.0,
                0)
        if data['mode'] == 'img2img':
            switch_mode = 0
            buffer = BytesIO(base64.b64decode(data['initimage']['image']))
            initimg = Image.open(buffer)
            # only positional arguments allowed because of *args  (CSV)
            fill_mode=data['inpainting_fill']

            oimages, oinfo, ohtml = img2img(switch_mode,data['prompt'],data['neg_prompt'],"","",initimg, "",
                        initimg,"", 0, data["steps"], 
                        smp_index, data["mask_blur"],              # mode and blur
                        fill_mode, 
                        data['restore_faces'], data['tiling'],
                        data['batch_count'],1,
                        data['cfg_scale'], data['denoising_strength'],
                        data['seed'],
                        0,0.0, 0, 0,False,
                        data['height'], data['width'],
                        0,  data["inpaint_full_res"],data["upscale_overlap"],data["inpainting_mask_invert"],
                        "","",0)
        if data['mode'] == 'inpainting':

            buffer = BytesIO(base64.b64decode(data['initimage']['image']))
            initimg = Image.open(buffer)
            initmask=None
            initmaskB64=data["initimage"]["mask"]
            if  initmaskB64:
                print("Mask included")
                buffer = BytesIO(base64.b64decode(data['initimage']['mask']))
                initmask = Image.open(buffer)
        
            fill_mode=data['inpainting_fill']

            # experimental: if mode is g-diffusion switch to orginal and use g-diffusion image as init image
            # image as init_image
            if (fill_mode==4):
                if (512, 512) != initimg.size and fill_mode==4: # default size is native img size
                    print("Inpainting: Resizing input img to 512x512 ")    
                    initimg = initimg.resize((512, 512), resample=PIL.Image.LANCZOS)
                init_image, initmask=gdiffusion.get_init_image(initimg,0.99,1, 0)
                fill_mode=1 # original
                initimg=init_image            
            else:
                if not initmask:
                    print("Generate mask from alpha")
                    initmask=gdiffusion.getAlphaAsImage(initimg)    # mask is now generated on server

            if (not initmask):
                if (gdiffusion.maskError==1):
                    print("inpainting: No transparent pixels found - throwing error")
                    return jsonify({'error':1,'text':"No transparent pixels found in image"})
                else:
                    print("inpainting: No  pixels found - throwing error")
                    return jsonify({'error':2,'text':"No pixels found in image"})                

            switch_mode = 1
            print("Fill:",fill_mode)

#def img2img(mode: int, prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, init_img, init_img_with_mask,
#  init_img_inpaint, init_mask_inpaint, mask_mode, steps: int,
#  sampler_index: int, mask_blur: int,
#  inpainting_fill: int, 
# restore_faces: bool, tiling: bool, 
# n_iter: int, batch_size: int, 
# cfg_scale: float, denoising_strength: float,
#  seed: int,
#  subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, 
# height: int, width: int
# resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, 
# img2img_batch_input_dir: str, img2img_batch_output_dir: str, *args):




            oimages, oinfo, ohtml = img2img(switch_mode,data['prompt'],data['neg_prompt'],"","",initimg, {'image':initimg, 'mask':initmask},
                        initimg,initmask, 1, data["steps"], 
                        smp_index, data["mask_blur"],              # mode and blur
                        fill_mode, 
                        data['restore_faces'], data['tiling'],
                        data['batch_count'],1,
                        data['cfg_scale'], data['denoising_strength'],
                        data['seed'],
                        0,0.0, 0, 0,False,
                        data['height'], data['width'],
                        0,  data["inpaint_full_res"],data["upscale_overlap"],data["inpainting_mask_invert"],
                        "","",0)


        b64images = []
        for img in oimages:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            b64images.append(img_str.decode())
        return {'images':b64images,'info':oinfo}

    #TODO this keeps the thread running outside gradio, but I think it breaks colab maybe? should work locally
    try:
      while True:
        time.sleep(0.1)
    except (KeyboardInterrupt, OSError):
        print("Keyboard interruption in main thread... closing server.")
        app.close()


if __name__ == "__main__":
    webui()
