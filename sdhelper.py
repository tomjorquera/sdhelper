"""
Helper library to experiment with Image generation models, preferably in notebooks.
"""
import base64
import hashlib
import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from PIL.Image import Image as ImageObject
from torch import autocast


@dataclass
class ResultItem:
    """An item among several produced by the evaluation of a model.

    ResultItem contains some outputs, but also all the inputs and parameters required to produce them.
    Depending on the model and how it was called, outputs can be things such as an image, multiple images, or more complex objects.
    """

    inputs: Any
    outputs: List[Any]
    params: Dict[str, Any]
    model_type: str
    model_params: Dict[str, Any]


class Result:
    """Result produced by evaluation of a model.

    A Result is an object containing some `sdhelper.ResultItem`, as well as useful metadata.

    The internals of a Result are *not* considered stable. To interact with it, use one of its methods (such as `to_pandas`).
    """

    def __init__(self, results: Dict[str, Any], meta: Dict[str, Any]) -> None:
        self.items = results
        self.meta = meta

    def _repr_html_(self):
        buffered = BytesIO()
        self.to_img_grid().save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return ("""<img src="data:image/png;base64,{0}" />""").format(img_str)

    def to_img_grid(
        self, rows: Optional[int] = None, cols: Optional[int] = None
    ) -> ImageObject:
        """Convert to a image grid.

        If rows or columns are not specified, will try to use hints from metadata, or good defaults.
        Note: this method is the method used for the html representation in jupyter notebooks.

        Args:
          rows:
            number of rows for the image grid (optional).
          cols:
            number of columns for the image grid (optional).

        Returns:
          An image grid composed of the Result images.
        """
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

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a pandas dataframe.

        Returns:
          A `pandas.DataFrame` representation of the result.

        """
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

    def save(self, output_dir: str) -> None:
        """Save the Result to a directory.

        Args:
          output_dir: the path of directory where to save the result.
        """
        data = self.to_pandas()

        output_path = Path(f"{output_dir}_{datetime.now().isoformat()}")
        info_path = output_path / "info"

        output_path.mkdir(parents=True, exist_ok=True)
        info_path.mkdir(parents=True, exist_ok=True)

        def img_to_digest(img: ImageObject) -> Optional[str]:
            if img is None:
                return None
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

    def merge_with(self, other: "Result") -> "Result":
        """Create a new Result combining this one and another.

        Create a new Result containing `sdhelper.ResultItem` from both sources.

        The original Results are not modified.

        This does not copy the metadata of either source.

        Args:
          other:
            the other Result to combine with.
        """
        res = Result([], {})
        res.items.extend(self.items)
        res.items.extend(other.items)
        return res


class SDModel(ABC):
    """Base class to implement a new model."""

    @abstractmethod
    def model_type(self) -> str:
        """String representation of the model type.

        This method should be overridden depending on the actual model result.

        Returns:
          a string representation of the model type.
        """
        pass

    @abstractmethod
    def model_parameters(self) -> Dict[str, Any]:
        """Parameters names and values of the model.

        This method should be overridden depending on the actual model result.

        Returns:
          A Dict containing the parameters names and values of the model.
        """
        pass

    @abstractmethod
    def _eval(
        self,
        model_input: Any,
        seed: int,
        guidance_scale: float,
        inference_steps: int,
        output_resolution: int = 512,
    ) -> Any:
        """Internal method to evaluate the model.

        This method should be overridden depending on the actual model result.

        This method should *not* be called directly. Use the `evaluate` method instead.

        Args:
          model_input:
            the inputs to give for the model evaluation.
          seed:
            the seed for randomization.
          guidance_scale:
            guidance scale parameter value.
          inference_steps:
            number of inference steps to run.
          output_resolution:
            resolution of the output (default: 512).

        Returns:
          some model output.
        """
        pass

    def evaluate(
        self,
        model_input: Any,
        seed: int,
        guidance_scale: float,
        inference_steps: int,
        output_resolution: int = 512,
    ) -> Result:
        """Model evaluation.

        Args:
          model_input:
            the inputs to give for the model evaluation.
          seed:
            the seed for randomization.
          guidance_scale:
            guidance scale parameter value.
          inference_steps:
            number of inference steps to run.
          output_resolution:
            resolution of the output (default: 512).

        Returns:
          A `sdhelper.Result` object
        """
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

    def _txt_prompt_from_input(self, model_input: Any) -> Optional[str]:
        """Get the text prompt part of the input (if any).

        This method should be overridden depending on the actual model result.

        Args:
          the raw model inputs

        Returns:
          the text prompt part of the model input (if any).
        """
        return None

    def _img_prompt_from_input(self, model_input: Any) -> Optional[ImageObject]:
        """Get the image prompt part of the input (if any).

        This method should be overridden depending on the actual model result.

        Args:
          the raw model inputs

        Returns:
          the image prompt part of the model input (if any).
        """
        return None

    def _img_result_from_output(self, model_output: Any) -> Optional[List[ImageObject]]:
        """Get the generated image part of the output (if any).

        This method should be overridden depending on the actual model result.

        Args:
          the raw model outputs

        Returns:
          the generated images part of the model outputs (if any).
        """
        return None

    def explore(
        self,
        model_input: Any,
        nb_images: int = 16,
        starting_seed: int = 0,
        guidance_scale: float = 4,
        inference_steps: int = 20,
        output_resolution: int = 512,
    ) -> Result:
        """Run multiple evaluations of the same input with varying seeds.

        Args:
          model_input:
            the inputs to give for the model evaluation.
          nb_images:
            number of images to generate.
          starting_seed:
            initial seed to use (default: 0).
          guidance_scale:
            guidance scale parameter value>
          inference_steps:
            number of inference steps to run.
          output_resolution:
            resolution of the output (default: 512).

        Returns:
          A `sdhelper.Result` object containing generated images for all seeds.
        """
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
        model_input: Any,
        seed: int,
        starting_steps: int = 20,
        output_resolution: int = 512,
    ) -> Result:
        """Run multiple evaluations of the same input with varying inference steps and guidance scale.

        Args:
          model_input:
            the inputs to give for the model evaluation.
          seed:
            seed to use.
          starting_steps:
            number of inference steps to start with (default: 20).
          output_resolution:
            resolution of the output (default: 512).

        Returns:
          A `sdhelper.Result` object containing generated images for all seeds.
        """
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
        model_input: Any,
        prompt_variants: Dict[str, List[str]],
        inference_steps: int = 20,
        seed: int = 0,
        guidance_scale: float = 7,
        output_resolution: int = 512,
    ) -> Result:
        """Run the model on a combination of multiple prompts.

        This method allows to easily test variations of outputs, by operating string replacement
        on a base prompt.

        TODO example

        Note: this is a simple string replacement, so be careful about the replacement keys you use.

        Args:
          model_input:
            the inputs to give for the model evaluation.
          prompt_variants:
            The dict of the values to replace and replacements.
          inference_steps:
            number of inference steps to run (default: 20).
          seed:
            seed to use (default: 0).
          guidance_scale:
            the guidance scale to use (default: 7).
          output_resolution:
            resolution of the output (default: 512).

        Returns:
          A `sdhelper.Result` object containing generated images for all combinations.
        """
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
    """SDModel implementation for HuggingFace model."""

    def __init__(
        self, token: str, version: str = "1-4", allow_nsfw: bool = False
    ) -> None:
        """
        Instanciate a new model using weights from HuggingFace.

        Note: you still need to register and accept HuggingFace use conditions.

        Args:
          token:
            your HuggingFace token.
          version:
            the exact model version to use (default: 1-4).
          allow_nsfw:
            allow NSFW images or not (default: false).

        """
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

    def model_type(self) -> str:
        return "HHFDiffuser"

    def model_parameters(self) -> Dict[str, Any]:
        return {"version": self.version, "allow_nsfw": self.allow_nsfw}

    def _txt_prompt_from_input(self, model_input: str) -> str:
        return model_input

    def _img_result_from_output(
        self, model_output: List[ImageObject]
    ) -> List[ImageObject]:
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
