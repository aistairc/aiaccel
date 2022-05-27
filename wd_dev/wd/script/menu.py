# -*- coding: utf-8 -*-

import curses
import string
import unicodedata
import datetime


class Menu(object):

    def __init__(self, menu_data):
        self.curses_start()
        self.menu = menu_data
        self.menu_select = [self.menu]
        '''
menuのサンプル。
(
  ('menu1',
    (
      ('menu1-1', self.sample),
      ('menu1-2', self.sample),
    )
  ),
  ('menu2', self.sample),
  ('menu3',
    (
      ('menu3-1', self.sample),  # 項目が1つの時は最後に必ず,でtupleに。
    )
  )
)
'''

    def curses_start(self):
        self.status = True
        self.cs = curses.initscr()  # curses standard screen
        curses.noecho()
        curses.cbreak()
        self.cs.keypad(True)
        self.height, self.width = self.cs.getmaxyx()
        
    def curses_end(self):  # curses.wrapper使用時は実行する必要無し。
        if self.status is None:
            return
        self.status = None;
        self.cs.keypad(False)
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    def mp(self, n, s1):
        self.p('{}.{}'.format(n, s1))

    def pr(self, s):
        self.p(s)
        self.r()

    def p(self, s):
        self.cs.addstr(s)

    def r(self):
        self.cs.refresh()

    def k(self, s=None):
        if type(s) is str:
            self.pr(s)
        return self.cs.getkey()

    def clear(self):
        self.cs.clear()

    def message(self, s):
        self.clear()
        self.p(s)
        self.k('\n任意のキーで次へ')

    def message_yesno(self, s):
        self.clear()
        self.p(s+'\n')
        self.mp(1, 'はい\n')
        self.mp('その他のキー', 'いいえ\n')
        self.r()
        ch = self.k()
        if ch == '1':
            return True
        else:
            return False

    def message_yesnocancel(self, s):
        self.clear()
        self.p(s)
        self.mp(1, 'はい\n')
        self.mp(2, 'いいえ\n')
        self.mp('その他のキー', 'キャンセル\n')
        self.r()
        ch = self.k()
        if ch == '1':
            return 1
        elif ch == '2':
            return 0
        else:
            return -1

    def draw(self):
        self.clear()
        cm = self.menu_select[-1]  # current menu
        if cm is self.menu:
            self.mp(0, 'プログラムを終了\n')
        else:
            self.mp(0, '前のメニューに戻る\n')
        for i, v in enumerate(cm):
            self.mp(i+1, v[0]+'\n')
        ch = self.k('数字を入力 ')
        if ch not in string.digits:
            return None
        n = int(ch)
        if n == 0:
            if cm is self.menu:
                return 'end'
            self.menu_select.pop()
            return None
        elif n <= len(cm):
            d = cm[n-1][1]
            if type(d) is tuple:
                self.menu_select.append(d)
            else:
                d()
        return None

    def wlen(self, s):
        n = 0
        for c in s:
            if unicodedata.east_asian_width(c) in 'WFA':
                n += 2
            else:
                n += 1
        return n

    def scroll_win(self, str, top_one_line=''):
        try:
            self.scroll_win_sub(str, top_one_line)
        except:
            self.message('''
申しわけありません。
大きなファイルには、まだ対応していません。

ファイル名: %s
おおよその大きさ: %d
''' % (top_one_line, len(str)))
    
    def scroll_win_sub(self, str, top_one_line=''):
        dy = self.height-2
        dx = self.width-4
        lines = str.split('\n')
        nline = len(lines)
        h = nline
        if nline < self.height:
            h = self.height
        max_width = max([self.wlen(line) for line in lines])+4
        p = curses.newpad(h, max_width)
        for i, v in enumerate(lines):
            p.addstr(i, 0, '%03d|%s' % (i, v))
        y = 0
        x = 0
        while True:
            self.clear()
            self.p(top_one_line+'\n')
            self.mp(1, '前へ ')
            self.mp(2, '次へ ')
            self.mp(3, '最後へ ')
            self.mp(4, '左へ ')
            self.mp(5, '右へ ')
            self.mp(0, '戻る')
            self.r()
            p.refresh(y, x,   2, 0,  self.height-1, self.width-1)
            ch = self.k()
            if ch == '0':
                break
            elif ch == '1':
                y -= dy
                if y < 0:
                    y = 0
            elif ch == '2':
                y += dy
                if y >= nline:
                    y -= dy
            elif ch == '3':
                y = nline-dy
            elif ch == '4':
                x -= dx
                if x < 0:
                    x = 0
            elif ch == '5':
                x += dx
                if x >= max_width:
                    x -= dx

    def filer(self, cp):
        si = 0
        while True:
            di = 7
            paths = list(cp.iterdir())
            paths.sort()
            self.clear()
            h = self.height
            self.p('8.前へ 9.次へ 0.戻る %s\n' % str(cp))
            for i, v in enumerate(paths):
                if i < si:
                    continue
                dt = datetime.datetime.fromtimestamp(v.stat().st_ctime)
                st = dt.strftime('%d-%H:%M:%S %Y/%m')
                if i >= si and i < si+di:
                    self.p(str(i-si+1)+'.')
                else:
                    self.p('  ')
                self.p('%s\t\t%s\n' % (v.name, st))
                if i-si+1 >= h-2:
                    break
            ch = self.k('数字を入力 ')
            if ch not in string.digits:
                continue
            n = int(ch)
            if n == 0:
                break
            elif n == 8:
                if si-di >= 0:
                    si -= di
            elif n == 9:
                if si+di < len(paths):
                    si += di
            elif n <= i-si+1:
                p = cp/paths[n+si-1]
                if p.is_dir():
                    self.filer(p)
                elif p.is_file():
                    self.scroll_win(p.read_text(), str(p))
