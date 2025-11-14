from typing import Any, TypeVar

from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813
from rich.logging import RichHandler
from rich.progress import track

import torch

import attrs

from aiaccel.config import overwrite_omegaconf_dumper, print_config

overwrite_omegaconf_dumper()

logger = logging.getLogger(__name__)


@attrs.define
class BasePipeline(metaclass=ABCMeta):
    """
    Base class for inference pipelines.

    .. note::
        Note that this class is an experimental feature and may change in the future.

    Basic usage:

    .. code-block:: python
        :caption: separate.py

        from pathlib import Path

        import attrs

        import torch

        from aiaccel.torch.pipelines import BasePipeline, reorder_fields
        from aiaccel.torch.lightning import load_checkpoint

        import soundfile as sf


        @attrs.define(field_transformer=reorder_fields)
        class SeparationPipeline(BasePipeline):
            checkpoint_path: str

            device: str = "cuda"

            src_ext: str = "wav"
            dst_ext: str = "wav"

            def __attrs_post_init__(self):
                self.model, self.config = load_checkpoint(self.checkpoint_path)
                self.model.eval()
                self.model.to(self.device)

            def __call__(self, wav: torch.Tensor) -> torch.Tensor:
                return self.model(wav)

            @torch.inference_mode()
            def inference_one(self, src_filename: Path, dst_filename: Path) -> None:
                wav_mix, sr = sf.load(src_filename, dtype="float32")
                assert sr == self.config.sr, f"Sample rate mismatch: {sr} != {self.config.sr}"

                wav_mix = torch.from_numpy(wav_mix).unsqueeze(0).to(self.device)
                wav_sep = self(wav_mix).squeeze(0).cpu().numpy()

                sf.write(dst_filename, wav_sep, sr)


        if __name__ == "__main__":
            SeparationPipeline.main()

    .. code-block:: bash

        python separate.py one --help


    .. code-block:: text

        usage: test.py one [-h] [--device DEVICE] [--src_ext SRC_EXT] [--dst_ext DST_EXT] [--allow_tf32] [--unk_args UNK_ARGS] checkpoint_path src_filename dst_filename

        positional arguments:
        checkpoint_path
        src_filename
        dst_filename

        options:
        -h, --help           show this help message and exit
        --device DEVICE
        --src_ext SRC_EXT
        --dst_ext DST_EXT
        --allow_tf32
        --unk_args UNK_ARGS

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

    def __post_init__(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        logger.info(f"Set torch.backends.cuda.matmul.allow_tf32 to {self.allow_tf32}")

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def inference_one(self, src_filename: Path, dst_filename: Path) -> Any:
        dst_filename.parent.mkdir(exist_ok=True, parents=True)

    @classmethod
    def main(cls) -> None:
        logging.basicConfig(level="NOTSET", handlers=[RichHandler(level="NOTSET", omit_repeated_times=False)])

        args, unk_args = cls._parse_arguments()

        print("=" * 32)
        print_config(oc.create(vars(args) | {"unk_args": unk_args}))
        print("=" * 32)

        kwargs = {field.name: getattr(args, field.name) for field in attrs.fields_dict(cls).values()}
        pipeline = cls(**kwargs, unk_args=unk_args or [])

        match args.cmd:
            case "one":
                pipeline.inference_one(args.src_filename, args.dst_filename)
            case "batch":
                src_fname_list = list(args.src_path.glob(f"*.{args.src_ext}"))
                src_fname_list.sort()

                if "TASK_INDEX" in os.environ:
                    start = int(os.environ["TASK_INDEX"]) - 1
                    end = start + int(os.environ["TASK_STEPSIZE"])

                    src_fname_list = src_fname_list[start:end]

                for src_filename in track(src_fname_list):
                    pipeline.inference_one(src_filename, args.dst_path / f"{src_filename.stem}.{args.dst_ext}")

    @classmethod
    def _parse_arguments(cls) -> tuple[Namespace, list[str]]:
        parent_parser = ArgumentParser(add_help=False)
        for fld in attrs.fields_dict(cls).values():
            if fld.init is False:
                pass
            elif fld.type is bool and fld.default is False:
                parent_parser.add_argument(f"--{fld.name}", action="store_true")
            elif fld.default is attrs.NOTHING:
                assert fld.type is not None
                parent_parser.add_argument(fld.name, type=fld.type)
            else:
                assert fld.type is not None
                parent_parser.add_argument(f"--{fld.name}", type=fld.type, default=fld.default)

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
            exit()

        return args, unk_args


T = TypeVar("T")


def reorder_fields(cls: Any, fields: list[attrs.Attribute[T]]) -> list[attrs.Attribute[T]]:
    """
    Reorder attrs fields such that fields without default values come first,
    followed by fields with default values, and inherited fields are placed last.
    """
    original_order = {fld: idx for idx, fld in enumerate(fields)}

    def sort_key(a: attrs.Attribute[T]) -> tuple[int, int, int]:
        inherited = 1 if a.inherited else 0  # type: ignore[attr-defined]  # TODO
        has_default = 1 if a.default is not attrs.NOTHING else 0

        return (has_default, inherited, original_order[a])

    return sorted(fields, key=sort_key)
