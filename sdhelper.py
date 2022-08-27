import math
import base64
import itertools
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List

import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from IPython.display import HTML
from PIL import Image
from torch import autocast


@dataclass
class ResultItem:
    inputs: Any
    output: List[Any]
    params: Dict[str, Any]
    model_type: str
    model_params: Dict[str, Any]


class Result:
    def __init__(self, results, meta):
        self.items = results
        self.meta = meta

    def _repr_html_(self):
        buffered = BytesIO()
        self.to_img_grid().save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return ("""<img src="data:image/png;base64,{0}" />""").format(img_str)

        return self.to_img_grid()

    def to_img_grid(self, rows=None, cols=None):
        imgs = [img for item in self.items for img in item.output]
        nb_images = len(imgs)

        # set grid hint for missing #rows/#cols values if any
        if "grid_hint" in self.meta:
            if rows is None and "rows" in self.meta["grid_hint"]:
                rows = self.meta["grid_hint"]["rows"]
            if cols is None and "cols" in self.meta["grid_hint"]:
                cols = self.meta["grid_hint"]["cols"]

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

    def merge_with(self, other):
        self.items.extend(other.items)


class SDModel:
    def model_type(self):
        pass

    def model_parameters(self):
        pass

    def _eval(
        self, model_input, seed, guidance_scale, inference_steps, output_resolution=512
    ):
        pass

    def evaluate(
        self, model_input, seed, guidance_scale, inference_steps, output_resolution=512
    ):
        res = self._eval(
            model_input, seed, guidance_scale, inference_steps, output_resolution
        )
        return Result(
            [
                ResultItem(
                    output=res,
                    inputs={
                        "txt_prompt": self._txt_prompt_from_input(model_input),
                        "img_prompt": self._img_prompt_from_input(model_input),
                    },
                    params={
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "inference_steps": inference_steps,
                        "output_solution": output_resolution,
                    },
                    model_type=self.model_type(),
                    model_params=self.model_parameters(),
                )
            ],
            meta={},
        )

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
        results = None
        for i in range(nb_images):
            seed = (
                starting_seed + i
            )  # + 2**(i) # TODO allow to configure seed increment
            result = self.evaluate(
                model_input,
                seed,
                guidance_scale,
                inference_steps,
                output_resolution,
            )
            if results is None:
                results = result
            else:
                results.merge_with(result)

        results.meta["grid_hint"] = {
            "rows": math.floor(math.sqrt(nb_images)),
            "cols": math.ceil(nb_images / math.floor(math.sqrt(nb_images))),
        }
        return results

    def exploit(
        self,
        model_input,
        seed,
        starting_steps=20,
        output_resolution=512,
    ):
        results = None
        # TODO make inference increments and guidance scale parameters
        for inference_steps in range(
            starting_steps, starting_steps * 7, starting_steps * 2
        ):
            for guidance_scale in range(4, 10, 2):
                result = self.evaluate(
                    model_input,
                    seed,
                    guidance_scale,
                    inference_steps,
                    output_resolution,
                )
                if results is None:
                    results = result
                else:
                    results.merge_with(result)

        results.meta[
            "grid_hint"
        ] = {  # TODO adjust hints once we can parametrize further
            "rows": 3,
            "cols": 3,
        }
        return results

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
        results = None
        for combinaison in keys_combinaisons:
            prompt = prompt_base
            for key, value in combinaison:
                prompt = prompt.replace(key, value)
                result = self.evaluate(
                    prompt, seed, guidance_scale, inference_steps, resolution
                )
            if results is None:
                results = result
            else:
                results.merge_with(result)

        rows = len(prompt_variants[list(prompt_variants)[0]])
        results.meta[
            "grid_hint"
        ] = {  # TODO adjust hints once we can parametrize further
            "rows": rows,
            "cols": int(len(keys_combinaisons) / rows),
        }
        return results


class HFDiffuser(SDModel):
    def __init__(self, token, version="1-4", allow_nsfw=False):
        self.version = version
        self.allow_nsfw = allow_nsfw

        self.pipe = StableDiffusionPipeline.from_pretrained(
            f"CompVis/stable-diffusion-v{version}",
            use_auth_token=token,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")

        if allow_nsfw:
            self.pipe.safety_checker = lambda images, **kwargs: (images, False)

    def model_type(self):
        return "HHFDiffuser"

    def model_parameters(self):
        return {"version": self.version, "allow_nsfw": self.allow_nsfw}

    def _txt_prompt_from_input(self, model_input):
        return model_input

    def _eval(
        self, model_input, seed, guidance_scale, inference_steps, output_resolution=512
    ):
        prompt = model_input
        with autocast("cuda", dtype=torch.float16):
            return self.pipe(
                prompt,
                height=output_resolution,
                width=output_resolution,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cuda").manual_seed(seed),
            )["sample"]
