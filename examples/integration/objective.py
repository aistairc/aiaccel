import subprocess

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(lr: float) -> float:
    # torch.app.train task.optimizer_config.optimizer_generator.lr=lr を実行
    subprocess.run(
        [
            "python",
            "-m",
            "aiaccel.torch.apps.train",
            "resnet50/config.yaml",
            f"task.optimizer_config.optimizer_generator.lr={lr}",
            f"trainer.logger.name=lr_{lr}",
        ]
    )

    # val loss を取得

    # ログディレクトリのパス
    log_dir = f"resnet50/lr_{lr}"

    # EventAccumulator で .tfevents ファイルを読み込む
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    val_loss_events = event_acc.Scalars("validation/loss")

    last_val_loss = float(val_loss_events[-1].value)

    return last_val_loss
