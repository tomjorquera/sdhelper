import base64
import hashlib
import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from IPython.display import HTML
from PIL import Image
from torch import autocast


@dataclass
class ResultItem:
    inputs: Any
    outputs: List[Any]
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

    def to_img_grid(self, rows=None, cols=None):
        imgs = [img for item in self.items for img in item.outputs["img"]]
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

    def to_pandas(self):
        # collect all params, model params
        params = set()
        model_params = set()
        for item in self.items:
            params.update(item.params.keys())
            model_params.update(item.model_params.keys())

        columns = [
            "outputs_img",
            "inputs_txt",
            "inputs_img",
            "model_type",
        ]

        columns.extend([f"model_param.{model_param}" for model_param in model_params])
        columns.extend([f"param.{param}" for param in params])

        elements = []
        for item in self.items:
            element = [
                item.outputs["img"][0]
                if len(item.outputs["img"]) == 1
                else item.outputs["img"],
                item.inputs["txt"],
                item.inputs["img"],
                item.model_type,
            ]
            for model_param in model_params:
                element.append(
                    item.model_params[model_param]
                    if model_param in item.model_params
                    else None,
                )
            for param in params:
                element.append(
                    item.params[param] if param in item.params else None,
                )
            elements.append(element)

        return pd.DataFrame(elements, columns=columns)

    def save(self, output_dir):
        data = self.to_pandas()

        output_path = Path(f"{output_dir}_{datetime.now().isoformat()}")
        info_path = output_path / "info"

        output_path.mkdir(parents=True, exist_ok=True)
        info_path.mkdir(parents=True, exist_ok=True)

        def img_to_digest(img):
            if img is None:
                return
            md5hash = hashlib.md5(img.tobytes())
            return md5hash.hexdigest()

        seen_imgs = {}

        input_imgs_col = []
        output_imgs_col = []

        for item in data.itertuples():
            # export input and output images
            if item.inputs_img:
                input_img = item.inputs_img
                img_hash = img_to_digest(input_img)
                if img_hash not in seen_imgs:
                    seen_imgs[img_hash] = f"image{len(seen_imgs)}.png"
                    input_img.save(info_path / seen_imgs[img_hash])
                input_imgs_col.append(info_path / seen_imgs[img_hash])
            else:
                input_imgs_col.append(None)

            if item.outputs_img:
                output_imgs = item.outputs_img
                output_names = []
                if not isinstance(output_imgs, list):
                    output_imgs = [output_imgs]
                for i, output_img in enumerate(output_imgs):
                    output_name = f"output_{item.Index}_{i}.png"
                    output_names.append(output_name)
                    output_img.save(output_path / output_name)
                if len(output_names) == 1:
                    output_names = output_names[0]
                output_imgs_col.append(output_names)
            else:
                output_imgs_col.append([])

        # replace images with references to the files
        data["inputs_img"] = input_imgs_col
        data["outputs_img"] = output_imgs_col

        # That's all folks
        data.to_csv(info_path / "data.csv")

    def merge_with(self, other):
        res = Result([], {})
        res.items.extend(self.items)
        res.items.extend(other.items)
        return res


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
                    outputs={"img": self._img_result_from_output(res)},
                    inputs={
                        "txt": self._txt_prompt_from_input(model_input),
                        "img": self._img_prompt_from_input(model_input),
                    },
                    params={
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "inference_steps": inference_steps,
                        "output_resolution": output_resolution,
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

    def _img_result_from_output(self, model_output):
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
                results = results.merge_with(result)

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
                    results = results.merge_with(result)

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
                results = results.merge_with(result)

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

    def _img_result_from_output(self, model_output):
        return model_output

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
