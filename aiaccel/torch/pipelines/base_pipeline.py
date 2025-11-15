from typing import TYPE_CHECKING, Any

from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
import logging
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf as oc  # noqa: N813
from rich.logging import RichHandler
from rich.progress import track

import torch

import attrs

from aiaccel.config import overwrite_omegaconf_dumper, print_config

overwrite_omegaconf_dumper()

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    Attribute = attrs.Attribute[Any]
else:
    Attribute = attrs.Attribute


@attrs.define()
class BasePipeline(metaclass=ABCMeta):
    """Base class for inference pipelines.

    .. note::

        Note that this class is an experimental feature and may change in the future.

    Basic usage:

    .. code-block:: python
        :caption: separate.py

        from typing import Any

        from argparse import ArgumentParser
        from pathlib import Path

        from omegaconf import OmegaConf as oc  # noqa: N813

        import torch

        import attrs

        import soundfile as sf

        from aiaccel.torch.lightning import load_checkpoint
        from aiaccel.torch.pipelines import BasePipeline, reorder_fields


        @attrs.define(slots=False, field_transformer=reorder_fields)
        class SeparationPipeline(BasePipeline):
            checkpoint_path: Path

            device: str = "cuda"

            src_ext: str = "wav"
            dst_ext: str = "wav"

            overwrite_config: dict[str, Any] | None = None

            def setup(self) -> None:
                self.model, self.config = load_checkpoint(
                    self.checkpoint_path,
                    device=self.device,
                    overwrite_config=self.overwrite_config,
                )
                self.model.eval()

            def __call__(self, wav: torch.Tensor) -> torch.Tensor:
                return self.model(wav)

            @torch.inference_mode()
            def process_one(self, src_filename: Path, dst_filename: Path) -> None:
                wav_mix, sr = sf.load(src_filename, dtype="float32")
                assert sr == self.config.sr, f"Sample rate mismatch: {sr} != {self.config.sr}"

                wav_mix = torch.from_numpy(wav_mix).unsqueeze(0).to(self.device)
                wav_sep = self(wav_mix).squeeze(0).cpu().numpy()

                sf.write(dst_filename, wav_sep, sr)

            @classmethod
            def _prepare_parser(cls, fields: list[attrs.Attribute]) -> ArgumentParser:
                return super()._prepare_parser(
                    list(filter(lambda f: f.name != "overwrite_config", fields))
                )

            @classmethod
            def _process_unk_args(
                cls, unk_args: list[str], kwargs: dict[str, Any], parser: ArgumentParser
            ) -> dict[str, Any]:
                return kwargs | {"overwrite_config": oc.from_cli(unk_args)}


        if __name__ == "__main__":
            SeparationPipeline.main()

    .. code-block:: bash

        python separate.py one --help

    .. code-block:: text

        usage: test.py one [-h] [--device DEVICE] [--src_ext SRC_EXT] [--dst_ext DST_EXT] [--allow_tf32] checkpoint_path src_filename dst_filename

        positional arguments:
        checkpoint_path
        src_filename
        dst_filename

        options:
        -h, --help         show this help message and exit
        --device DEVICE
        --src_ext SRC_EXT
        --dst_ext DST_EXT
        --allow_tf32

    .. code-block:: bash

        # run inference for one file
        python separate.py one ./mixture.wav ./result.wav --checkpoint_path=./sepformer/

        # run inference for all files in a directory
        python separate.py batch ./mixtures/ ./results/ --checkpoint_path=./sepformer/

    """  # noqa: E501

    src_ext: str = attrs.field(init=False)
    dst_ext: str = attrs.field(init=False)
    allow_tf32: bool = False
    unk_args: list[str] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        logger.info(f"Set torch.backends.cuda.matmul.allow_tf32 to {self.allow_tf32}")

        self.setup()

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def process_one(self, src_filename: Path, dst_filename: Path) -> Any:
        pass

    @classmethod
    def main(cls) -> None:
        logging.basicConfig(level="NOTSET", handlers=[RichHandler(level="NOTSET", omit_repeated_times=False)])

        parent_parser = cls._prepare_parser(attrs.fields(cls))
        args, unk_args, parser = cls._parse_arguments(parent_parser)

        kwargs = {fld.name: getattr(args, fld.name) for fld in attrs.fields(cls) if hasattr(args, fld.name)}
        kwargs = cls._process_unk_args(unk_args, kwargs, parser)

        print_config(oc.create(vars(args) | {"unk_args": unk_args}))

        pipeline = cls(**kwargs)

        match args.cmd:
            case "one":
                args.dst_filename.parent.mkdir(exist_ok=True, parents=True)
                pipeline.process_one(args.src_filename, args.dst_filename)
            case "batch":
                src_fname_list = list(args.src_path.glob(f"*.{args.src_ext}"))
                src_fname_list.sort()

                if "TASK_INDEX" in os.environ:
                    start = int(os.environ["TASK_INDEX"]) - 1
                    end = start + int(os.environ["TASK_STEPSIZE"])

                    src_fname_list = src_fname_list[start:end]

                args.dst_path.mkdir(exist_ok=True, parents=True)
                for src_filename in track(src_fname_list):
                    pipeline.process_one(src_filename, args.dst_path / f"{src_filename.stem}.{args.dst_ext}")

    @classmethod
    def _prepare_parser(cls, fields: list[Attribute]) -> ArgumentParser:
        parser = ArgumentParser(add_help=False)

        for fld in fields:
            if fld.init is False or fld.name == "unk_args":
                pass
            elif fld.default is attrs.NOTHING:
                assert fld.type is not None
                parser.add_argument(fld.name, type=fld.type)
            elif fld.type is bool:
                if fld.default is False:
                    parser.add_argument(f"--{fld.name}", action="store_true")
                elif fld.default is True:
                    parser.add_argument(f"--no-{fld.name}", action="store_false", dest=fld.name)
            else:
                assert fld.type is not None
                parser.add_argument(f"--{fld.name}", type=fld.type, default=fld.default)

        return parser

    @classmethod
    def _parse_arguments(cls, parent_parser: ArgumentParser) -> tuple[Namespace, list[str], ArgumentParser]:
        parser = ArgumentParser()
        sub_parsers = parser.add_subparsers(dest="cmd")

        sub_parser = sub_parsers.add_parser("one", parents=[parent_parser])
        sub_parser.add_argument("src_filename", type=Path)
        sub_parser.add_argument("dst_filename", type=Path)

        sub_parser = sub_parsers.add_parser("batch", parents=[parent_parser])
        sub_parser.add_argument("src_path", type=Path)
        sub_parser.add_argument("dst_path", type=Path)

        args, unk_args = parser.parse_known_args()

        if args.cmd is None:
            parser.print_help()
            sys.exit(0)

        return args, unk_args, parser

    @classmethod
    def _process_unk_args(cls, unk_args: list[str], kwargs: dict[str, Any], parser: ArgumentParser) -> dict[str, Any]:
        if len(unk_args) > 0:
            parser.parse_args()  # to show help message

        return kwargs


def reorder_fields(cls: Any, fields: list[Attribute]) -> list[Attribute]:
    """
    Reorder attrs fields such that fields without default values come first, then fields with default values.
    They are further ordered such that fields defined in the class come before inherited fields.

    Basic usage:

    .. code-block:: python

        import attrs
        from aiaccel.torch.pipelines import reorder_fields


        @attrs.define(field_transformer=reorder_fields)
        class MyPipeline(BasePipeline):
            required_field: int
            optional_field: str = "default"

    """
    original_order = {fld: idx for idx, fld in enumerate(fields)}

    def sort_key(a: Attribute) -> tuple[int, int, int]:
        inherited = 1 if a.inherited else 0  # type: ignore[attr-defined]  # TODO
        has_default = 1 if a.default is not attrs.NOTHING else 0

        return (has_default, inherited, original_order[a])

    return sorted(fields, key=sort_key)
