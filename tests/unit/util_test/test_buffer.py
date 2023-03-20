from aiaccel.util.buffer import Buffer


def test_set_max_len():
    buff = Buffer(["test"])

    assert buff.d["test"]._max_size == 65535
    buff.d["test"].set_max_len(255)
    assert buff.d["test"]._max_size == 255


def test_Add():
    buff = Buffer(["test"])
    print(buff.d["test"])
    assert buff.d["test"].Data == []
    buff.Add("test", 1)
    assert buff.d["test"].Data[0] == 1
    assert buff.d["test"](0) == 1


def test_Add_lengthover():
    buff = Buffer(["test"])
    buff.d["test"].set_max_len(2)
    assert buff.d["test"]._max_size == 2

    buff.Add("test", 1)
    buff.Add("test", 2)
    assert buff.d["test"].Len == 2
    assert buff.d["test"](0) == 1
    assert buff.d["test"](1) == 2

    buff.Add("test", 3)
    assert buff.d["test"].Len == 2
    assert buff.d["test"](0) == 2
    assert buff.d["test"](1) == 3

    buff.Add("test", 4)
    assert buff.d["test"].Len == 2
    assert buff.d["test"](0) == 3
    assert buff.d["test"](1) == 4


def test_Pre():
    buff = Buffer(["test"])
    buff.Add("test", 6)
    assert buff.d["test"].Pre is None
    buff.Add("test", 7)
    assert buff.d["test"].Pre == 6


def test_Now():
    buff = Buffer(["test"])
    buff.Add("test", 4)
    assert buff.d["test"].Now == 4


def test_Del():
    buff = Buffer(["test"])
    buff.Add("test", 1)
    assert buff.d["test"](0) == 1
    assert buff.d["test"].Len == 1
    buff.Del("test", 0)
    assert buff.d["test"].Len == 0


def test_Clear():
    buff = Buffer(["test"])
    buff.Add("test", 1)
    assert buff.d["test"](0) == 1
    assert buff.d["test"].Len == 1
    buff.Clear("test")
    assert buff.d["test"].Len == 0


def test_replace():
    buff = Buffer(["test"])
    buff.Add("test", 1)
    buff.Add("test", 2)
    buff.Add("test", 3)
    buff.Add("test", 4)
    buff.Add("test", 5)
    assert buff.d["test"].Data == [1, 2, 3, 4, 5]
    new_arr = [6, 7, 8, 8, 9]
    buff.d["test"].Replace(new_arr)
    assert buff.d["test"].Data == [6, 7, 8, 8, 9]


def test_Value():
    buff = Buffer(["test"])
    buff.Add("test", 1)
    buff.Add("test", 2)
    assert buff.d["test"].Value(0) == 1
    assert buff.d["test"].Value(1) == 2


def test_Is_Empty():
    buff = Buffer(["test"])
    assert buff.d["test"].Is_Empty is True
    buff.Add("test", 1)
    assert buff.d["test"].Is_Empty is False


def test_Duplicate():
    buff = Buffer(["test"])
    buff.Add("test", 1)
    buff.Add("test", 2)
    buff.Add("test", 3)
    assert buff.d["test"].Duplicate(1) == 0
    assert buff.d["test"].Duplicate(2) == 1
    assert buff.d["test"].Duplicate(3) == 2
    assert buff.d["test"].Duplicate(4) == -1


def test_delta():
    buff = Buffer(["test"])
    buff.Add("test", 1)
    buff.Add("test", 5)
    assert buff.d["test"].delta() == 4


def test_point_diff():
    buff = Buffer(["test"])
    buff.Add("test", 1)
    buff.Add("test", 5)
    assert buff.d["test"].point_diff(0, 1) == 4


def test_has_difference():
    buff = Buffer(["test"])
    assert buff.d["test"].has_difference() == False

    buff.Add("test", 1.12)
    buff.Add("test", 5.45)
    assert buff.d["test"].has_difference() == True
    assert buff.d["test"].has_difference(digit=1) == True

    buff.Add("test", 1.12)
    buff.Add("test", 1.12)
    assert buff.d["test"].has_difference() == False
    assert buff.d["test"].has_difference(digit=1) == False

    buff.Add("test", 1.12)
    buff.Add("test", 1.13)
    assert buff.d["test"].has_difference() == True
    assert buff.d["test"].has_difference(digit=1) == False
