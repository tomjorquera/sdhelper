import math
import itertools

from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


class SDModel:
    def evaluate(
        self, model_input, seed, guidance_scale, inference_steps, output_resolution=512
    ):
        pass

    def _txt_prompt_from_input(self, model_input):
        return None

    def _img_prompt_from_input(self, model_input):
        return None

    def explore(
        self,
        model_input,
        nb_images=16,
        starting_seed=0,
        guidance_scale=4,
        inference_steps=20,
        output_resolution=512,
    ):

        result_items = []
        for i in range(nb_images):
            seed = (
                starting_seed + i
            )  # + 2**(i) # TODO allow to configure seed increment
            result_items.extend(
                self.evaluate(
                    model_input,
                    seed,
                    guidance_scale,
                    inference_steps,
                    output_resolution,
                )["items"]
            )

        return {
            "items": result_items,
            "meta": {
                "grid_hint": {
                    "rows": math.floor(math.sqrt(nb_images)),
                    "cols": math.ceil(nb_images / math.floor(math.sqrt(nb_images))),
                }
            },
        }

    def exploit(
        self,
        model_input,
        seed,
        starting_steps=20,
        output_resolution=512,
    ):
        result_items = []
        # TODO make inference increments and guidance scale parameters
        for inference_steps in range(
            starting_steps, starting_steps * 7, starting_steps * 2
        ):
            for guidance_scale in range(4, 10, 2):
                result_items.extend(
                    self.evaluate(
                        model_input,
                        seed,
                        guidance_scale,
                        inference_steps,
                        output_resolution,
                    )["items"]
                )

        return {
            "items": result_items,
            "meta": {
                "grid_hint": {  # TODO adjust hints once we can parametrize further
                    "rows": 3,
                    "cols": 3,
                }
            },
        }

    def combine(
        self,
        model_input,
        prompt_variants,
        inference_steps=20,
        seed=0,
        guidance_scale=7,
        resolution=512,
    ):
        prompt_base = self._txt_prompt_from_input(model_input)
        keys_combinaisons = [
            [[k, v] for v in prompt_variants[k]] for k in prompt_variants
        ]
        keys_combinaisons = list(itertools.product(*keys_combinaisons))
        result_items = []
        for combinaison in keys_combinaisons:
            prompt = prompt_base
            for key, value in combinaison:
                prompt = prompt.replace(key, value)
            result_items.extend(
                self.evaluate(
                    prompt, seed, guidance_scale, inference_steps, resolution
                )["items"]
            )

        rows = len(prompt_variants[list(prompt_variants)[0]])
        return {
            "items": result_items,
            "meta": {
                "grid_hint": {
                    "rows": rows,
                    "cols": int(len(keys_combinaisons) / rows),
                }
            },
        }


class Diffuser(SDModel):
    def __init__(self, token, version="1-4", allow_nsfw=False):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            f"CompVis/stable-diffusion-v{version}",
            use_auth_token=token,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")

        if allow_nsfw:
            self.pipe.safety_checker = lambda images, **kwargs: (images, False)

    def _txt_prompt_from_input(self, model_input):
        return model_input

    def evaluate(
        self, model_input, seed, guidance_scale, inference_steps, output_resolution=512
    ):
        prompt = model_input
        with autocast("cuda", dtype=torch.float16):
            out_image = self.pipe(
                prompt,
                height=output_resolution,
                width=output_resolution,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cuda").manual_seed(seed),
            )["sample"]
            return {
                "items": [
                    {
                        "output": out_image,
                        "input": {
                            "text_prompt": prompt,
                            "seed": seed,
                            "guidance_scale": guidance_scale,
                            "inference_steps": inference_steps,
                            "output_solution": output_resolution,
                        },
                    }
                ],
                "meta": {},
            }


def result_to_img_grid(result, rows=None, cols=None):
    imgs = [img for res in result["items"] for img in res["output"]]
    nb_images = len(imgs)

    # set grid hint for missing #rows/#cols values if any
    if "grid_hint" in result["meta"]:
        if rows is None and "rows" in result["meta"]["grid_hint"]:
            rows = result["meta"]["grid_hint"]["rows"]
        if cols is None and "cols" in result["meta"]["grid_hint"]:
            cols = result["meta"]["grid_hint"]["cols"]

    # compute missing #rows/#cols values
    if rows is None and cols is None:
        rows = math.floor(math.sqrt(nb_images))
        cols = math.ceil(nb_images / math.floor(math.sqrt(nb_images)))
    elif rows is None:
        rows = math.ceil(nb_images / math.floor(cols))
    elif cols is None:
        cols = math.ceil(nb_images / math.floor(rows))

    assert nb_images <= rows * cols

    width, height = imgs[0].size  # TODO check *max* among images
    grid = Image.new("RGB", size=(cols * width, rows * height))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * width, i // cols * height))

    # TODO print ascii schema with more infos?
    return grid
