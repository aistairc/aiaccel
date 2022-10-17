import os

import aiaccel
from aiaccel.abci.qsub import create_qsub_command
from aiaccel.config import load_config


def test_create_qsub_command(load_test_config):
    config = load_test_config()
    optimizer_file = os.path.join(
        os.path.join('', aiaccel.dict_runner),
        ""
    )
    qsub_command = create_qsub_command(config, optimizer_file)
    assert type(qsub_command) is list


# # --------------------------------------------------------------
# # ↓　手動確用(コメント解除)

# def test_create_qsub_command__additionaloptions():
#     # 追加オプションあり(一つ)
#     config = load_config("./additionaloptions.json")
#     optimizer_file = os.path.join(
#         os.path.join('', aiaccel.dict_runner),
#         config.get('optimizer', 'optimizer_runner')
#     )
#     qsub_command = create_qsub_command(config, optimizer_file)
#     print(qsub_command)
#     assert type(qsub_command) is list


# def test_create_qsub_command__additionaloptions2():
#     # 追加オプションあり(複数)
#     config = load_config("./additionaloptions2.json")
#     optimizer_file = os.path.join(
#         os.path.join('', aiaccel.dict_runner),
#         config.get('optimizer', 'optimizer_runner')
#     )
#     qsub_command = create_qsub_command(config, optimizer_file)
#     print(qsub_command)
#     assert type(qsub_command) is list


# def test_create_qsub_command__no_additionaloptions():
#     config = load_config("./config.json")
#     optimizer_file = os.path.join(
#         os.path.join('', aiaccel.dict_runner),
#         config.get('optimizer', 'optimizer_runner')
#     )
#     qsub_command = create_qsub_command(config, optimizer_file)
#     print(qsub_command)
#     assert type(qsub_command) is list


# def test_create_qsub_command__no_additionaloptions_2():
#     config = load_config("./config2.json")
#     optimizer_file = os.path.join(
#         os.path.join('', aiaccel.dict_runner),
#         config.get('optimizer', 'optimizer_runner')
#     )
#     qsub_command = create_qsub_command(config, optimizer_file)
#     print(qsub_command)
#     assert type(qsub_command) is list


# def test_create_qsub_command__no_additionaloptions_3():
#     config = load_config("./config3.json")
#     optimizer_file = os.path.join(
#         os.path.join('', aiaccel.dict_runner),
#         config.get('optimizer', 'optimizer_runner')
#     )
#     qsub_command = create_qsub_command(config, optimizer_file)
#     print(qsub_command)
#     assert type(qsub_command) is list


# if __name__ == "__main__":
#     test_create_qsub_command__additionaloptions()
#     test_create_qsub_command__additionaloptions2()
#     test_create_qsub_command__no_additionaloptions()
#     test_create_qsub_command__no_additionaloptions_2()
#     test_create_qsub_command__no_additionaloptions_3()
