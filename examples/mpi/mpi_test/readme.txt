2023.04.23

abciのosがRocky Linux release 8.6 (Green Obsidian)に変更さた。
mpi moduleがopenmpi/4.1.3からhpcx-mt/2.12に変更された。
従来のままではaiaccelのmpiが動作せず。
原因が分からず。
基本的なテストプログラムを作成。
念のため、その時のログを残す。
aiaccelのmpiがhpcx-mt/2.12に対応した後は必要無いが。
---

ssh abci
qrsh -g your_group_id -l rt_C.small=1 -l h_rt=2:00:00
module load python/3.11/3.11.2
module load hpcx-mt/2.12
cd ~/mpi_work
python3 -m venv mpienv
source mpienv/bin/activate
pip install --upgrade pip
pip install mpi4py
# git clone github:aistairc/aiaccel.git
# 本document作成時はまだmainにmergeしていないので
cp -r my_development_dir/aiaccel ~/mpi_work
exit
cd ~/mpi_work/aiaccel/examples/experimental/mpi/mpi_test
qsub -g your_group_id q02.sh
# 約20秒後に。
qstat
# jobがなくなっていることを確認。
# abciが作成したjobの出力結果を確認。
cat q02.sh.o39627509
rank=0 processor=g0008.abci.local size=5
before sleep(20)
rank=3 processor=g0010.abci.local n=13 sleep(10)
rank=4 processor=g0010.abci.local n=14 sleep(10)
rank=2 processor=g0008.abci.local n=12 sleep(10)
rank=1 processor=g0008.abci.local n=11 sleep(10)
end main()
---

以上
