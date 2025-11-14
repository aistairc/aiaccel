from typing import Any, Generic, TypeVar
from typing_extensions import Self

from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import MISSING, dataclass, field, fields
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813
from rich.logging import RichHandler
from rich.progress import track

from aiaccel.config import overwrite_omegaconf_dumper, print_config

overwrite_omegaconf_dumper()


@dataclass
class BaseConfig:
    unk_args: list[str] = field(default_factory=list)

    @classmethod
    def from_namespace(cls, args: Namespace, unk_args: list[str] | None = None) -> Self:
        return cls(**{field.name: getattr(args, field.name) for field in fields(cls)}, unk_args=unk_args or [])

    @classmethod
    def get_argument_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(add_help=False)
        for fld in fields(cls):
            if fld.name == "unk_args":
                pass
            elif fld.type is bool and fld.default is False:
                parser.add_argument(f"--{fld.name}", action="store_true")
            elif fld.default is MISSING:
                parser.add_argument(fld.name, type=fld.type)
            else:
                parser.add_argument(f"--{fld.name}", type=fld.type, default=fld.default)

        return parser


C = TypeVar("C", bound=BaseConfig)


class BasePipeline(Generic[C], metaclass=ABCMeta):
    default_src_ext: str
    default_dst_ext: str

    allow_tf32: bool | None = False

    Config: type[C]

    def __init__(self, *, config: C | None = None) -> None:
        if config is None:
            config = self.Config()

        self.config = config

        if self.allow_tf32 is not None:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32

    @abstractmethod
    def inference_one(self, src_filename: Path, dst_filename: Path) -> Any:
        dst_filename.parent.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @classmethod
    def main(cls) -> None:
        logging.basicConfig(level="NOTSET", handlers=[RichHandler(level="NOTSET", omit_repeated_times=False)])

        args, unk_args = cls._parse_arguments()

        print("=" * 32)
        print_config(oc.create(vars(args) | {"unk_args": unk_args}))
        print("=" * 32)

        config = cls.Config.from_namespace(args, unk_args)
        pipeline = cls(config=config)

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
        parent_parser = cls.Config.get_argument_parser()

        parser = ArgumentParser()
        sub_parsers = parser.add_subparsers(dest="cmd")

        sub_parser = sub_parsers.add_parser("one", parents=[parent_parser])
        sub_parser.add_argument("src_filename", type=Path)
        sub_parser.add_argument("dst_filename", type=Path)

        sub_parser = sub_parsers.add_parser("batch", parents=[parent_parser])
        sub_parser.add_argument("src_path", type=Path)
        sub_parser.add_argument("dst_path", type=Path)
        sub_parser.add_argument("--src_ext", type=str, default=cls.default_src_ext)
        sub_parser.add_argument("--dst_ext", type=str, default=cls.default_dst_ext)

        args, unk_args = parser.parse_known_args()

        if args.cmd is None:
            parser.print_help()
            exit()

        return args, unk_args
