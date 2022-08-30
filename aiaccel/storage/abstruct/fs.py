# import aiaccel
# from aiaccel.storage.fs.datalist import Datalist
# from pathlib import PosixPath


# class Abstruct:
#     def __init__(self, workspace: PosixPath, dir_name: str) -> None:
#         self.workspace = workspace
#         self.path = self.workspace / dir_name
#         self.datas = Datalist()
#         self.file_type = "*.hp"

#     def select_table(self, tabale_name: str):
#         pass

#     def get_file_list(self) -> list:
#         return list(self.path.glob(self.file_type))

#     def update(self):
#         self.datas.clear()
#         paths = self.get_file_list()
#         for path in paths:
#             trial_id = int(path.stem)
#             self.datas.add(trial_id, _jobstate(self.workspace, trial_id))
