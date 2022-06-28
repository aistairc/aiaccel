# -*- coding: utf-8-unix -*-

import subprocess
import re
import time

from aiaccel.config import load_config

import wd
from wd.base import AbstractBase
from wd.script.menu import Menu


class Cui(AbstractBase):

    def __init__(self, config):
        super().__init__(config)

    def init(self):
        self.logger = wd.set_logger(
            'root.cui', 'DEBUG',
            self.dict_log/wd.cui_log_file,
            'Cui      ')
        self.menu_data = (
            ('aiaccelの実行',
             (('実行方法の簡単な説明', self.opt_help),
              ('計算ノード数の確認', self.opt_calc_node_num),
              ('実行', self.opt_do))),
            ('workの閲覧など',
             (('*-wd/work閲覧。treeで', self.view_wd_work_by_tree),
              ('*-wd/work/log閲覧', self.view_wd_work_log),
              ('*-wd/work閲覧', self.view_wd_work),
              ('aiaccel閲覧', self.view_opt))),
            ('コマンド(ps,qstat,qdel-all,nreq-ls,nvidia-smiなど)',
             (('ps', self.ps),
              ('qstat', self.qstat),
              ('qdel all', self.qdel_all),
              ('nreq nvidia-smi.py q0000', self.nreq_nvidia_smi),
              ('nreq rsync.py q0000', self.nreq_rsync),
              ('nreq ps.py q0000', self.nreq_ps),
              ('nreq ls.py q0000', self.nreq_ls))),
            ('wd.bin.qsubの実行、停止、動作確認',
             (('動作確認', self.check_qsub),
              ('実行', self.start_qsub),
              ('停止', self.stop_qsub))),
            ('*-wd/workの削除',
             (('状態保存ファイルとログファイルを残してその他は全て削除',
               self.clear_work_wo_var_log),
              ('ログファイルのみ残してその他は全て削除',
               self.clear_work_wo_log),
              ('全てのファイルを削除', self.clear_work_all))),

        )
        self.menu = Menu(self.menu_data)
        self.logger.info('start')

    def base_main(self):
        self.init()
        try:
            while True:
                self.sub_main()
        except:
            self.menu.curses_end()
            pass

    def sub_main(self):
        ret = self.menu.draw()
        if ret is not None:
            if ret == 'end':
                self.normal_exit(ret)
            else:
                self.error_exit(ret)

    def check_qsub(self):
        if self.path_qsub.is_file():
            self.menu.message('qsubは実行中です。')
            return True
        else:
            self.menu.message('qsubは停止しています。')
            return False

    def start_qsub(self):
        if self.path_qsub.is_file():
            self.menu.message('qsubは実行中です。'
                              '何もしないで、終了します。')
            return
        cmd = '(cd %s; python -m wd.bin.qsub &)' % self.path_config
        self.logger.debug('start qsub')
        subprocess.Popen(cmd, shell=True)
        self.menu.message('qsubを実行しました。')

    def stop_qsub(self):
        if not self.path_qsub.is_file():
            self.menu.message('qsubは停止中です。'
                              '何もしないで、終了します。')
            return
        self.logger.debug('stop qsub')
        self.path_qsub.unlink()
        self.menu.message('qsubを終了しました。')

    def remove_sub_dirs(self, path, _include):
        include = [str(p) for p in _include]
        a = []
        for p in path.glob('**'):
            if str(p.parent) in include:
                a.append(p)
        for p in a:
            p.unlink()

    def remove_only_files(self, path, _exclude=[]):
        exclude = [str(p) for p in _exclude]
        for p in path.glob('**/*'):
            if p.is_file() and str(p.parent) not in exclude:
                p.unlink()

    def clear_work_wo_var_log(self):
        s = '''
危険な操作です。
%s
フォルダの状態保存ファイルとログファイル以外全てを削除します。
よろしいですか。
''' % self.ws
        if self.menu.message_yesno(s):
            self.remove_only_files(
                self.ws,
                [self.dict_var, self.dict_log])
            self.remove_sub_dirs(
                self.ws,
                [self.dict_gpu, self.dict_node])
            s = (str(self.ws)+'\nフォルダの状態保存ファイルと'
                 'ログファイル以外全てを削除しました。')
            self.menu.message(s)
        else:
            self.menu.message('何もしませんでした。')

    def clear_work_wo_log(self):
        s = '''
危険な操作です。
%s
フォルダのログファイル以外全てを削除します。
よろしいですか。
''' % self.ws
        if self.menu.message_yesno(s):
            self.remove_only_files(self.ws, [self.dict_log])
            self.remove_sub_dirs(self.ws, [self.dict_gpu, self.dict_node])
            s = (str(self.ws)+'\nフォルダのログファイル以外全てを'
                 '削除しました。')
            self.menu.message(s)
        else:
            self.menu.message('何もしませんでした。')

    def clear_work_all(self):
        s = '''
危険な操作です。
%s
フォルダの全てのファイルを削除します。
ログファイルも削除するため、削除実行後、
本プログラムを終了します。
よろしいですか。
''' % self.ws
        if self.menu.message_yesno(s):
            self.remove_only_files(self.ws)
            s = (str(self.ws)+'\nフォルダの全てのファイルを削除しました。\n'
                 'プログラムを終了します。')
            self.menu.message(s)
            self.normal_exit()
        else:
            self.menu.message('何もしませんでした。')

    def view_wd_work(self):
        self.menu.filer(self.ws)

    def view_wd_work_by_tree(self):
        cmd = '(cd %s; tree -an)' % self.ws
        proc = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        res = proc.stdout.decode('utf8')
        self.menu.scroll_win(res, cmd)

    def view_wd_work_log(self):
        self.menu.filer(self.dict_log)
        
    def view_opt(self):
        n = self.menu.filer(self.ws.joinpath('../../../..').resolve())

    def ps(self):
        self.exec_cmd('ps -fe | grep `whoami`')

    def qstat(self):
        self.exec_cmd('qstat')

    def qdel_all(self):
        proc = subprocess.run(
            'qstat', shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        res = proc.stdout.decode('utf8')
        if len(res) == 0:
            self.menu.message('qdel_all: qstat: 標準出力は空でした。')
            return
        ret = ''
        p = r'\s+(\d+).+q00'
        for line in res.split('\n'):
            m = re.match(p, line)
            if m is not None:
                subprocess.run('qdel '+m.groups()[0], shell=True)
        self.qstat()
        
    def nreq_ls(self, qn='q0000'):
        cmd = ('python -m wd.bin.nreq %s '
               '$WD_DEV_DIR/examples/nreq/ls.py' % qn)
        self.exec_cmd(cmd)

    def nreq_nvidia_smi(self, qn='q0000'):
        cmd = ('python -m wd.bin.nreq %s '
               '$WD_DEV_DIR/examples/nreq/nvidia-smi.py' % qn)
        self.exec_cmd(cmd)
        
    def nreq_ps(self, qn='q0000'):
        cmd = ('python -m wd.bin.nreq %s '
               '$WD_DEV_DIR/examples/nreq/ps.py' % qn)
        self.exec_cmd(cmd)
        
    def nreq_rsync(self, qn='q0000'):
        cmd = ('python -m wd.bin.nreq %s '
               '$WD_DEV_DIR/examples/nreq/rsync.py' % qn)
        self.exec_cmd(cmd)
        
    def exec_cmd(self, cmd):
        proc = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        res = proc.stdout.decode('utf8')
        if len(res) == 0:
            self.menu.message('%s: 標準出力は空でした。' % cmd)
            return
        self.menu.scroll_win(res, cmd)
        
    def opt_help(self):
        self.menu.message('''
メニューの2番から順に実行して下さい。

今のところ、中断後の再開には対応していません。
いくつかの機能を、作成中です。
''')

    def opt_calc_node_num(self):
        nopt, ntotal, ntry, nlarge, nsmall = self.calc_node_num()
        large = self.config.get('wd', 'max_large_minutes')
        small = self.config.get('wd', 'max_small_minutes')
        self.menu.message('''
aiaccelの数: %d
num_nodeの合計: %d
trial_numberの合計: %d
計算ノードの合計: %d
G.large: %d
G.small: %d

計算ノードの合計数が大きすぎるなどの妥当性のチェックはしていません。
大きすぎる場合は、aiaccelのnum_nodeで調整して下さい。

G.largeは、71時間で終了します。
G.smallは、167時間で終了します。
再実行は実装確認中です。
spotのG.largeは72時間で、G.smallは168時間で、abciから強制終了されます。

一時的にpointを消費するため、小規模なテストなどの時は、
*-wd/config.ymlの
wd.max_large_minutes 現在の値: %s
wd.max_small_minutes 現在の値: %s
で終了までの時間を短くすることができます。
''' % (nopt, ntotal, ntry, nlarge+nsmall, nlarge, nsmall, large, small))
        
    def opt_do(self):
        yes = self.menu.message_yesno('''
aiaccelの実行を開始します。

完全な初期状態で、実行して下さい。
今のところ、前回のデータの有無などのチェックはしていません。
そのため、データや実行中のプロセスなどがあると、誤動作します。
(チェック機能は、作成中です。)

よろしいですか。
''')
        if not yes:
            self.menu.message('中断します。')
            return
        nopt, ntotal, ntry, nlarge, nsmall = self.calc_node_num()
        self.menu.message('''
wd経由のaiaccelの計算ノードでの実行を開始後、
本プログラムを終了します。
必要な場合は、再度、
本プログラム(python -m wd.bin.cui)を実行して、
途中経過を確認できます。
''')
        self.menu.curses_end()
        self.curses_ended = True
        self.opt_do_qsub()
        self.opt_do_qreq(nlarge, nsmall)
        self.opt_do_nreq(nlarge, nsmall, nopt)
        print('''
aiaccelの計算ノードでの実行を開始しました。
途中経過は、再度、
python -m wd.bin.cui
を実行することにより確認できます。
そのため、abciからログアウトしていただいても問題有りません。
本プログラム(python -m wd.bin.cui)を終了します。!!!
''', flush=True)
        self.normal_exit()

    def opt_do_qsub(self):
        cmd = '(cd %s; python -m wd.bin.qsub &)' % self.path_config
        self.logger.debug('wd.bin.qsub')
        subprocess.run(cmd, shell=True)
        print('wd.bin.qsubを実行しました。', flush=True)

    def opt_do_qreq_wait_nd(self, n):
        qn = wd.qsub_fmt % n
        print('\n%sのnd待ち !!!\n' % qn, flush=True)
        p1 = self.dict_node/qn
        for cnt in range(20):
            print('.', end='', flush=True)
            time.sleep(3)
            if p1.is_dir():
                print('!', flush=True)
                break
        
    def opt_do_qreq(self, nlarge, nsmall):
        cnt = 0
        for n in range(nlarge):
            cmd = '(cd %s; python -m wd.bin.qreq)' % self.path_config
            self.logger.debug('wd.bin.qreq')
            subprocess.run(cmd, shell=True)
            self.opt_do_qreq_wait_nd(cnt)
            cnt += 1
        for n in range(nsmall):
            cmd = '(cd %s; python -m wd.bin.qreq -s)' % self.path_config
            self.logger.debug('wd.bin.qreq -s')
            subprocess.run(cmd, shell=True)
            self.opt_do_qreq_wait_nd(cnt)
            cnt += 1
        print('wd.bin.qreqを実行しました。', flush=True)

    def opt_do_nreq(self, nlarge, nsmall, nopt):
        for n in range(nlarge+nsmall):
            qs = wd.qsub_fmt % n
            if n == 0:
                cmd = ('(cd %s; python -m wd.bin.nreq %s '
                       '$WD_DEV_DIR/wd/bin/run_gpu.py)' %
                       (self.path_config, qs))
                self.logger.debug('wd.bin.nreq %s gpu' % qs)
                subprocess.run(cmd, shell=True)
                for n1 in range(nopt):
                    cmd = ('(cd %s; python -m wd.bin.nreq -o %d %s %s)' %
                           (self.path_config, n1, qs,
                            wd.file_run_aiaccel_py))
                    self.logger.debug('wd.bin.nreq -o %d %s opt' %
                                      (n1, qs))
                    subprocess.run(cmd, shell=True)
            cmd = ('(cd %s; python -m wd.bin.nreq %s '
                   '$WD_DEV_DIR/wd/bin/run_ai.py)' %
                   (self.path_config, qs))
            self.logger.debug('wd.bin.nreq %s ai' % qs)
            subprocess.run(cmd, shell=True)
        print('wd.bin.nreqを実行しました。', flush=True)
