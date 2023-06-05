"""時間関連。."""

import datetime


def get_now_str() -> str:
    """現在時刻の文字列を取得。.

    Returns:
        string: yyyymmdd-hhMMss-nnn という文字列を返す。nnnはmsec。
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def get_timestamp(now_str: str) -> float:
    """get_now_str()の文字列から時刻の文字列を取得。.

    Returns:
        float: timestampを返す。
    """
    dt = datetime.datetime.strptime(now_str + "000", "%Y%m%d-%H%M%S-%f")
    return dt.timestamp()
