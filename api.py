import os
import threading
import time
import importlib
import signal
import threading

from fastapi.middleware.gzip import GZipMiddleware

from modules.paths import script_path

from modules import devices, sd_samplers
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2imgapi
import modules.img2imgapi

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork
from io import BytesIO
from PIL import Image
import json
import base64
from typing import Union
from pydantic import BaseModel
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
    steps: int = 16
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
    resize_mode: int = 0
    upscaler: str = ''
    upscale_overlap: int = 64
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 32
    inpainting_mask_invert: int = 0 # should be bool
    restore_faces: bool = False
    tiling: bool = False

    def __getitem__(self, item):
        return getattr(self, item)

class apiInputPlusImage(apiInput):
    initimage: apiImage = None

modelloader.cleanup_models()
modules.sd_models.setup_model()
codeformer.setup_model(cmd_opts.codeformer_models_path)
gfpgan.setup_model(cmd_opts.gfpgan_models_path)
shared.face_restorers.append(modules.face_restoration.FaceRestoration())
modelloader.load_upscalers()
queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    def f(*args, **kwargs):
        devices.torch_gc()

        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.job_timestamp = shared.state.get_job_timestamp()
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0
        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.textinfo = None

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        devices.torch_gc()

        return res

    return modules.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)


modules.scripts.load_scripts()

shared.sd_model = modules.sd_models.load_model()
shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

def webui():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    while 1:

        demo = modules.ui.create_ui(wrap_gradio_gpu_call=wrap_gradio_gpu_call)
        
        (app, local, gradio_remote) = demo.launch(
            share=cmd_opts.share,
            server_name="0.0.0.0" if cmd_opts.listen else None,
            server_port=cmd_opts.port,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )
        app.add_middleware(GZipMiddleware, minimum_size=1000)
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


                oimages, oinfo, ohtml = modules.txt2imgapi.txt2img(0,
                            prompt= data['prompt'],
                            negative_prompt= data['neg_prompt'],
                            steps= data["steps"], 
                            samplerindex= smp_index,
                            restore_faces=data['restore_faces'],
                            tiling= data['tiling'],
                            cfg_scale= data['cfg_scale'],
                            seed= data['seed'],
                            height= data['height'],
                            width= data['width'])
            if data['mode'] == 'img2img':
                switch_mode = 0
                buffer = BytesIO(base64.b64decode(data['initimage']['image']))
                initimg = Image.open(buffer)
                # only positional arguments allowed because of *args  (CSV)
                fill_mode=data['inpainting_fill']

                oimages, oinfo, ohtml = modules.img2imgapi(initimg,"","","",0,0,
                            mode=switch_mode,
                            prompt= data['prompt'],
                            negative_prompt= data['neg_prompt'],
                            steps=data["steps"], 
                            samplerindex= smp_index,
                            mask_blur= data["mask_blur"],
                            inpainting_fill= fill_mode, 
                            restore_faces=data['restore_faces'],
                            tiling= data['tiling'],
                            cfg_scale= data['cfg_scale'],
                            denoising_strength= data['denoising_strength'],
                            seed= data['seed'],
                            height= data['height'],
                            width= data['width'],
                            upscale_overlap=data["upscale_overlap"])
            if data['mode'] == 'inpainting':

                buffer = BytesIO(base64.b64decode(data['initimage']['image']))
                initimg = Image.open(buffer)
                buffer = BytesIO(base64.b64decode(data['initimage']['mask']))
                initmask = Image.open(buffer)
                switch_mode = 1
                fill_mode=data['inpainting_fill']


                oimages, oinfo, ohtml = modules.img2imgapi.img2img(initimg,{'image':initimg, 'mask':initmask},initimg,initmask,1,0,
                            mode=switch_mode,
                            prompt= data['prompt'],
                            negative_prompt= data['neg_prompt'],
                            steps=data["steps"], 
                            samplerindex= smp_index,
                            mask_blur= data["mask_blur"],
                            inpainting_fill= fill_mode, 
                            restore_faces=data['restore_faces'],
                            tiling= data['tiling'],
                            cfg_scale= data['cfg_scale'],
                            denoising_strength= data['denoising_strength'],
                            seed= data['seed'],
                            height= data['height'],
                            width= data['width'], 
                            inpaint_full_res= data["inpaint_full_res"],
                            inpaint_full_res_padding= data["inpaint_full_res_padding"],
                            upscale_overlap=data["upscale_overlap"])


            b64images = []
            for img in oimages:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue())
                b64images.append(img_str.decode())
            return {'images':b64images,'info':oinfo}

        while 1:
            time.sleep(0.5)
            if getattr(demo, 'do_restart', False):
                time.sleep(0.5)
                demo.close()
                time.sleep(0.5)
                break

        sd_samplers.set_samplers()

        print('Reloading Custom Scripts')
        modules.scripts.reload_scripts(os.path.join(script_path, "scripts"))
        print('Reloading modules: modules.ui')
        importlib.reload(modules.ui)
        print('Restarting Gradio')



if __name__ == "__main__":
    webui()
