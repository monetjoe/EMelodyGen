# 对abc文本处理，达成转到15个调的小目标
# 为了方便，还是按小节处理
import os
import re


Key_list = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
Note_list = ['C', 'D', 'E', 'F', 'G', 'A',
             'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b']
Pitch_sign_list = ['_', '=', '^', '\'', ',']

Transpose_key_matrix = {    # 转调矩阵，左列key为原调，右侧为要转的调
    'Cb': ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#'],
    'Gb': ['Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab'],
    'Db': ['Db', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb'],
    'Ab': ['Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb'],
    'Eb': ['Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F'],
    'Bb': ['Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C'],
    'F': ['F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G'],
    'C': ['C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D'],
    'G': ['G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A'],
    'D': ['D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E'],
    'A': ['A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B'],
    'E': ['E',  'B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#'],
    'B': ['B',  'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#'],
    'F#': ['F#', 'C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab'],
    'C#': ['C#', 'Ab', 'Eb', 'Bb', 'F',  'C',  'G',  'D',  'A',  'E',  'B',  'F#', 'C#', 'Ab', 'Eb'],
}

Key_accidental_dict = {
    'Cb': {'C': '_', 'D': '_', 'E': '_', 'F': '_', 'G': '_', 'A': '_', 'B': '_'},
    'Gb': {'C': '_', 'D': '_', 'E': '_', 'F': '=', 'G': '_', 'A': '_', 'B': '_'},
    'Db': {'C': '=', 'D': '_', 'E': '_', 'F': '=', 'G': '_', 'A': '_', 'B': '_'},
    'Ab': {'C': '=', 'D': '_', 'E': '_', 'F': '=', 'G': '=', 'A': '_', 'B': '_'},
    'Eb': {'C': '=', 'D': '=', 'E': '_', 'F': '=', 'G': '=', 'A': '_', 'B': '_'},
    'Bb': {'C': '=', 'D': '=', 'E': '_', 'F': '=', 'G': '=', 'A': '=', 'B': '_'},
    'F': {'C': '=', 'D': '=', 'E': '=', 'F': '=', 'G': '=', 'A': '=', 'B': '_'},
    'C': {'C': '=', 'D': '=', 'E': '=', 'F': '=', 'G': '=', 'A': '=', 'B': '='},
    'G': {'C': '=', 'D': '=', 'E': '=', 'F': '^', 'G': '=', 'A': '=', 'B': '='},
    'D': {'C': '^', 'D': '=', 'E': '=', 'F': '^', 'G': '=', 'A': '=', 'B': '='},
    'A': {'C': '^', 'D': '=', 'E': '=', 'F': '^', 'G': '^', 'A': '=', 'B': '='},
    'E': {'C': '^', 'D': '^', 'E': '=', 'F': '^', 'G': '^', 'A': '=', 'B': '='},
    'B': {'C': '^', 'D': '^', 'E': '=', 'F': '^', 'G': '^', 'A': '^', 'B': '='},
    'F#': {'C': '^', 'D': '^', 'E': '^', 'F': '^', 'G': '^', 'A': '^', 'B': '='},
    'C#': {'C': '^', 'D': '^', 'E': '^', 'F': '^', 'G': '^', 'A': '^', 'B': '^'},
}

Key2index = {
    'Cb': 11, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6,
    'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11,
}
Note2index = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}

Note2pitch = {
    'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
    'c': 72, 'd': 74, 'e': 76, 'f': 77, 'g': 79, 'a': 81, 'b': 83,
    '__C': 58, '_C': 59, '=C': 60, '^C': 61, '^^C': 62,
    '__D': 60, '_D': 61, '=D': 62, '^D': 63, '^^D': 64,
    '__E': 62, '_E': 63, '=E': 64, '^E': 65, '^^E': 66,
    '__F': 63, '_F': 64, '=F': 65, '^F': 66, '^^F': 67,
    '__G': 65, '_G': 66, '=G': 67, '^G': 68, '^^G': 69,
    '__A': 67, '_A': 68, '=A': 69, '^A': 70, '^^A': 71,
    '__B': 69, '_B': 70, '=B': 71, '^B': 72, '^^B': 73,
    '__c': 70, '_c': 71, '=c': 72, '^c': 73, '^^c': 74,
    '__d': 72, '_d': 73, '=d': 74, '^d': 75, '^^d': 76,
    '__e': 74, '_e': 75, '=e': 76, '^e': 77, '^^e': 78,
    '__f': 75, '_f': 76, '=f': 77, '^f': 78, '^^f': 79,
    '__g': 77, '_g': 78, '=g': 79, '^g': 80, '^^g': 81,
    '__a': 79, '_a': 80, '=a': 81, '^a': 82, '^^a': 83,
    '__b': 81, '_b': 82, '=b': 83, '^b': 84, '^^b': 85,
}

Pitch2note = {
    0:  ['=',   '__',   None,   None,   None,   '^^^',  '^'],
    1:  ['^',   '_',    '___',  None,   None,   None,   '^^'],
    2:  ['^^',  '=',    '__',   '___',  None,   None,   '^^^'],
    3:  ['^^^', '^',    '_',    '__',   None,   None,   None,],
    4:  [None,  '^^',   '=',    '_',    '___',  None,   None,],
    5:  [None,  '^^^',  '^',    '=',    '__',   None,   None,],
    6:  [None,  None,   '^^',   '^',    '_',    '___',  None,],
    7:  [None,  None,   '^^^',  '^^',   '=',    '__',   None,],
    8:  [None,  None,   None,   '^^^',  '^',    '_',    '___',],
    9:  ['___', None,   None,   None,   '^^',   '=',    '__',],
    10: ['__',  None,   None,   None,   '^^^',  '^',    '_',],
    11: ['_',   '___',  None,   None,   None,   '^^',   '=',],
}

Pitch2Chordnote = {
    0:  ['',    'bb',   None,   None,   None,   None,   '#'],
    1:  ['#',   'b',    None,   None,   None,   None,   '##'],
    2:  ['##',  '',     'bb',   None,   None,   None,   None],
    3:  [None,  '#',    'b',    'bb',   None,   None,   None],
    4:  [None,  '##',   '',     'b',    None,   None,   None],
    5:  [None,  None,   '#',    '',     'bb',   None,   None],
    6:  [None,  None,   '##',   '#',    'b',    None,   None],
    7:  [None,  None,   None,   '##',   '',     'bb',   None],
    8:  [None,  None,   None,   None,   '#',    'b',    None],
    9:  [None,  None,   None,   None,   '##',   '',     'bb'],
    10: ['bb',  None,   None,   None,   None,   '#',    'b'],
    11: ['b',   None,   None,   None,   None,   '##',   '', ],
}


def find_all_abc(directory):
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith('.abc') or file_path.endswith('txt'):
                yield file_path


def lookup_new_key_to_transpose(new_key, ori_key, des_key):
    # 针对出现新调的情况，计算新调应该转到哪个调
    for i, key in enumerate(Transpose_key_matrix[ori_key]):
        if key == des_key:
            return Transpose_key_matrix[new_key][i]


def transpose_a_chordnote(chordnote, ori_key, des_key):
    # chordnote最多一个升降号，以#和b标记

    interval = (Key2index[des_key] - Key2index[ori_key] + 12) % 12
    # 计算绝对音高, C=0
    chordnotename = chordnote[0]
    accidental = chordnote[1:]

    absolute_pitch = Note2pitch[chordnotename] + \
        accidental.count('#') - accidental.count('b')
    transposed_absolute_pitch = (absolute_pitch + interval + 12) % 12

    # 计算转调后的音名和升降号
    transposed_chordnotename = Key_list[(Key_list.index(
        chordnotename.upper()) + Note2index[des_key[0]] - Note2index[ori_key[0]] + 7) % 7]
    transposed_accidental = Pitch2Chordnote[transposed_absolute_pitch %
                                            12][Note2index[transposed_chordnotename]]
    if transposed_accidental is None:
        raise Exception('Cannot find proper notename')
    elif len(transposed_accidental) >= 2:
        if transposed_accidental[0] == '#':
            transposed_chordnotename = Key_list[(
                Note2index[transposed_chordnotename] + 1) % 7]
        elif transposed_accidental[0] == 'b':
            transposed_chordnotename = Key_list[(
                Note2index[transposed_chordnotename] + 6) % 7]
        else:
            raise Exception('Cannot find proper notename')
        transposed_accidental = Pitch2note[transposed_absolute_pitch %
                                           12][Note2index[transposed_chordnotename]]

    transposed_chordnote = transposed_chordnotename + transposed_accidental

    return transposed_chordnote


def transpose_a_note(note, ori_key, des_key):
    # 计算 des_key 和 ori_key 的 interval，0<=interval<=6，上移，7<=interval<=11，下移
    interval = (Key2index[des_key] - Key2index[ori_key] + 12) % 12
    if 7 <= interval <= 11:
        interval = interval - 12

    # 计算绝对音高，C == 60
    pattern = r'([^A-Ga-g]*)([A-Ga-g])([^A-Ga-g]*)'
    match = re.findall(pattern, note)[0]
    accidental = match[0]
    notename = match[1]
    octave = match[2]
    absolute_pitch = Note2pitch[notename] + accidental.count('^') - accidental.count('_') \
        + octave.count(r"'") * 12 - octave.count(r",") * 12
    transposed_absolute_pitch = absolute_pitch + interval

    # 计算转调后的音名和升降号
    transposed_notename = Key_list[(Key_list.index(
        notename.upper()) + Note2index[des_key[0]] - Note2index[ori_key[0]] + 7) % 7]
    transposed_accidental = Pitch2note[transposed_absolute_pitch %
                                       12][Note2index[transposed_notename]]
    if transposed_accidental is None:
        raise Exception('Cannot find proper notename')
    elif len(transposed_accidental) >= 3:
        if transposed_accidental[0] == '^':
            transposed_notename = Key_list[(
                Note2index[transposed_notename] + 1) % 7]
        elif transposed_accidental[0] == '_':
            transposed_notename = Key_list[(
                Note2index[transposed_notename] + 6) % 7]
        else:
            raise Exception('Cannot find proper notename')
        transposed_accidental = Pitch2note[transposed_absolute_pitch %
                                           12][Note2index[transposed_notename]]

    # 根据 transposed_absolute_pitch 来推算字母大小写以及',情况
    transposed_octave = ""
    temp_pitch_upper = Note2pitch[transposed_accidental +
                                  transposed_notename.upper()]
    temp_pitch_lower = Note2pitch[transposed_accidental +
                                  transposed_notename.lower()]
    if transposed_absolute_pitch < temp_pitch_upper:
        transposed_notename = transposed_notename.upper()
        transposed_octave = (temp_pitch_upper -
                             transposed_absolute_pitch) // 12 * r","
    elif transposed_absolute_pitch == temp_pitch_upper:
        transposed_notename = transposed_notename.upper()
    elif transposed_absolute_pitch == temp_pitch_lower:
        transposed_notename = transposed_notename.lower()
    elif temp_pitch_lower < transposed_absolute_pitch:
        transposed_notename = transposed_notename.lower()
        transposed_octave = (transposed_absolute_pitch -
                             temp_pitch_lower) // 12 * r"'"
    else:
        raise Exception('Cannot find octave')

    transposed_note = transposed_accidental + \
        transposed_notename + transposed_octave

    return transposed_note


def transpose_a_voice(abc_text, ori_key, des_key):

    if ori_key == 'none':
        return abc_text

    exclaim_re = r'![^!]+!'
    quote_re = r'"[^"]+"'
    squareBracket_re = r'\[[^\]]+\]'

    barlines = ["|:", "::", ":|", "[|", "||", "|]", "|"]
    barline_re = '|'.join(re.escape(s) for s in barlines)

    # 表示每个ascii字符的转调方式。-1：未定；0：引号外区域，不转调；1：按音符转调；2.按和弦转调；3.按[K:]转调；-2：引号内容；4：小节线
    transpose_ascii_list = [-1] * len(abc_text)

    # 根据 ori_key 以及 [K:] 来判断所有ascii字符的所在调，并挑出所有按[K:]转调的元素
    ori_key_ascii_list = [ori_key] * len(abc_text)
    keynote_list = []
    keynote_start_index_list = []
    keynote_end_index_list = []

    squareBracket_matches = re.finditer(squareBracket_re, abc_text)
    for squareBracket_match in squareBracket_matches:
        sqaureBracket_start = squareBracket_match.start()
        sqaureBracket_end = squareBracket_match.end()
        squareBracket_string = squareBracket_match.group()
        if squareBracket_string[1:3] == 'K:' and squareBracket_string[3] in Key_list:
            key = squareBracket_string[3:-1]
            key_start = sqaureBracket_start + 3
            key_end = sqaureBracket_end - 1
            for i in range(key_start, len(ori_key_ascii_list)):
                ori_key_ascii_list[i] = key
            for i in range(key_start, key_end):
                transpose_ascii_list[i] = 3
            for j in range(sqaureBracket_start, key_start):
                transpose_ascii_list[j] = 0
            for j in range(key_end, sqaureBracket_end):
                transpose_ascii_list[j] = 0

            keynote_list.append(key)
            keynote_start_index_list.append(key_start)
            keynote_end_index_list.append(key_end)

        elif squareBracket_string[2] == ':':    # information field，全部置0
            for j in range(sqaureBracket_start, sqaureBracket_end):
                transpose_ascii_list[j] = 0
        else:   # chord，只将[]置0
            transpose_ascii_list[sqaureBracket_start] = 0
            transpose_ascii_list[sqaureBracket_end - 1] = 0

    # 挑出和弦符号
    chordnote_list = []
    chordnote_start_index_list = []
    chordnote_end_index_list = []

    quote_matches = re.finditer(quote_re, abc_text)
    for quote_match in quote_matches:
        quote_start = quote_match.start()
        quote_end = quote_match.end()
        quoted_string = quote_match.group()
        if quoted_string[1] in Key_list:
            chordnote_re = r'[A-G][b#]?'
            chordnote_matches = re.finditer(chordnote_re, quoted_string)
            for chordnote_match in chordnote_matches:
                chordnote_start = quote_start + chordnote_match.start()
                chordnote_end = quote_start + chordnote_match.end()
                chordnote = chordnote_match.group()
                for i in range(chordnote_start, chordnote_end):
                    transpose_ascii_list[i] = 2

                chordnote_list.append(chordnote)
                chordnote_start_index_list.append(chordnote_start)
                chordnote_end_index_list.append(chordnote_end)

            for j in range(quote_start, quote_end):  # 除 chordnote 外全部置-2
                if transpose_ascii_list[j] != 2:
                    transpose_ascii_list[j] = -2
        else:   # text annotations 全部置0
            for j in range(quote_start, quote_end):
                transpose_ascii_list[j] = -2

    # 挑出!!包裹的区域
    exclaim_matches = re.finditer(exclaim_re, abc_text)
    for exclaim_match in exclaim_matches:
        exclaim_start = exclaim_match.start()
        exclaim_end = exclaim_match.end()
        for j in range(exclaim_start, exclaim_end):
            transpose_ascii_list[j] = -2

    # 挑小节线
    barline_list = []
    barline_start_index_list = []
    barline_end_index_list = []

    barline_matches = re.finditer(barline_re, abc_text)
    for barline_match in barline_matches:
        barline_start = barline_match.start()
        barline_end = barline_match.end()
        barline_list.append(barline_match.group())
        barline_start_index_list.append(barline_start)
        barline_end_index_list.append(barline_end)
        for i in range(barline_start, barline_end):
            if transpose_ascii_list[i] != -2:
                transpose_ascii_list[i] = 4

    # 挑出音符
    note_section_string_list = []   # for debug
    note_list = []
    note_start_index_list = []
    note_end_index_list = []

    for i, char in enumerate(abc_text):
        if transpose_ascii_list[i] == -1:
            if char not in Note_list + Pitch_sign_list:
                transpose_ascii_list[i] = 0
            else:
                transpose_ascii_list[i] = 1

    # print(abc_text)

    i = 0
    note_section_string_list = []
    while i < len(transpose_ascii_list):
        if transpose_ascii_list[i] == 1:
            j = i + 1
            while j < len(transpose_ascii_list):
                if transpose_ascii_list[j] == 1:
                    j += 1
                else:
                    break
            note_section_string = abc_text[i:j]
            note_section_string_list.append(note_section_string)

            note_re = r'[=^_]*[A-Ga-g](?:\'*,*)'
            note_matches = re.finditer(note_re, note_section_string)
            for note_match in note_matches:
                note_start = note_match.start() + i
                note_end = note_match.end() + i
                note_string = note_match.group()
                note_list.append(note_string)
                note_start_index_list.append(note_start)
                note_end_index_list.append(note_end)

            i = j + 1
        else:
            i += 1

    des_key_ascii_list = []
    for key in ori_key_ascii_list:
        des_key_ascii_list.append(
            lookup_new_key_to_transpose(key, ori_key, des_key))

    elements_to_transpose = []
    for i in range(len(barline_list)):
        elements_to_transpose.append({
            'type': 4, 'note': barline_list[i],
            'start': barline_start_index_list[i], 'end': barline_end_index_list[i],
            'transposed_content': barline_list[i]
        })
    for i in range(len(keynote_list)):
        elements_to_transpose.append({
            'type': 3, 'note': keynote_list[i], 'start': keynote_start_index_list[i], 'end': keynote_end_index_list[i],
            'ori_key': ori_key_ascii_list[keynote_start_index_list[i]],
            'des_key': des_key_ascii_list[keynote_start_index_list[i]],
            'transposed_content': des_key_ascii_list[keynote_start_index_list[i]],
        })
    for i in range(len(chordnote_list)):
        elements_to_transpose.append({
            'type': 2, 'note': chordnote_list[i], 'start': chordnote_start_index_list[i], 'end': chordnote_end_index_list[i],
            'ori_key': ori_key_ascii_list[chordnote_start_index_list[i]],
            'des_key': des_key_ascii_list[chordnote_start_index_list[i]],
            'transposed_content': None
        })
    for i in range(len(note_list)):
        elements_to_transpose.append({
            'type': 1, 'note': note_list[i], 'start': note_start_index_list[i], 'end': note_end_index_list[i],
            'ori_key': ori_key_ascii_list[note_start_index_list[i]],
            'des_key': des_key_ascii_list[note_start_index_list[i]],
            'actual_note': None,
            'transposed_actual_note': None,
            'transposed_content': None
        })
    elements_to_transpose = sorted(
        elements_to_transpose, key=lambda x: x['start'])

    transposed_abc_text = ''
    index = 0
    for i, ele in enumerate(elements_to_transpose):
        transposed_abc_text += abc_text[index: ele['start']]
        index = ele['end']

        if ele['type'] == 3 or ele['type'] == 4:
            pass
        elif ele['type'] == 2:
            ele['transposed_content'] = transpose_a_chordnote(
                ele['note'], ele['ori_key'], ele['des_key'])
        elif ele['type'] == 1:
            if ele['note'][0] in ['^', '=', '_']:  # 如果本身带临时升降号，实际音高就为带临时升降号的音
                ele['actual_note'] = ele['note']
            else:   # 往前回溯到一个小节线/key/相同音名和八度的音符
                j = i

                while j > 0:
                    j -= 1
                    if elements_to_transpose[j]['type'] == 4 or elements_to_transpose[j]['type'] == 3:
                        # 根据调号来判断实际音高
                        note_name = ele['note'][0].upper()
                        ori_key = ele['ori_key']
                        ele['actual_note'] = Key_accidental_dict[ori_key][note_name] + ele['note']
                        break
                    elif elements_to_transpose[j]['type'] == 1:
                        # 若音和前一个相同（注意，不仅音名，八度也要相同）
                        pre_note = elements_to_transpose[j]['note']
                        # 去掉前面的临时升降号，但不去掉后面的八度记号
                        pre_note_name = re.sub(r'^[=^_]*', '', pre_note)
                        if ele['note'] == pre_note_name:
                            ele['actual_note'] = elements_to_transpose[j]['actual_note']
                            break
                        else:
                            # 如果音名和前一个不同，则继续往前回溯，直到碰到小节线/key/相同音高的音符
                            pass

                # 如果很不幸就是第一个元素，或者没有回溯到以上三种类型，则根据调号来判断实际音高
                if j == 0 and ele['actual_note'] is None:
                    note_name = ele['note'][0].upper()
                    ori_key = ele['ori_key']
                    ele['actual_note'] = Key_accidental_dict[ori_key][note_name] + ele['note']

            # 至此，所有音符的'actual_note'项都应该有值
            ele['transposed_actual_note'] = transpose_a_note(
                ele['actual_note'], ele['ori_key'], ele['des_key'])

            # 填 transposed_content，即判断是否去掉临时升降号。往前回溯到一个小节线/key/相同音名和八度的音符
            j = i
            pattern = r'([^A-Ga-g]*)([A-Ga-g])([^A-Ga-g]*)'
            match = re.findall(pattern, ele['transposed_actual_note'])[0]
            accidental = match[0]
            notename = match[1]
            octave = match[2]

            while j > 0:
                j -= 1
                # 回溯到小节线或者调号
                if elements_to_transpose[j]['type'] == 4 or elements_to_transpose[j]['type'] == 3:
                    # 根据调号来判断是否去掉临时升降号
                    des_key = ele['des_key']
                    # 和调式升降号相符，则去掉临时升降号
                    if accidental == Key_accidental_dict[des_key][notename.upper()]:
                        ele['transposed_content'] = notename + octave
                    else:
                        ele['transposed_content'] = ele['transposed_actual_note']
                    break
                elif elements_to_transpose[j]['type'] == 1:  # 回溯到音符
                    pre_note = elements_to_transpose[j]['transposed_actual_note']
                    pre_notename_octave = re.sub(r'^[=^_]*', '', pre_note)
                    cur_note = ele['transposed_actual_note']
                    cur_notename_octave = re.sub(r'^[=^_]*', '', cur_note)
                    if pre_notename_octave == cur_notename_octave:  # 如果transposed_actual_note的音名+八度和前一个相同
                        # 如果就是同一个音，则去掉临时升降号
                        if ele['transposed_actual_note'] == elements_to_transpose[j]['transposed_actual_note']:
                            ele['transposed_content'] = notename + octave
                        else:   # 如果不是同一个音，保留临时升降号
                            ele['transposed_content'] = ele['transposed_actual_note']
                        break
                    else:   # 继续往前回溯
                        pass

            # 如果很不幸就是第一个元素，或者没有回溯到以上三种类型，则根据调号来判断是否去掉临时升降号
            if j == 0 and ele['transposed_content'] is None:
                des_key = ele['des_key']
                # 和调式升降号相符，则去掉临时升降号
                if accidental == Key_accidental_dict[des_key][notename.upper()]:
                    ele['transposed_content'] = notename + octave
                else:
                    ele['transposed_content'] = ele['transposed_actual_note']

        transposed_abc_text += ele['transposed_content']

    transposed_abc_text += abc_text[index:]

    print('')
    return transposed_abc_text


def transpose_an_abc_text(abc_text_lines, des_key):

    reserved_info_field = ['L:', 'K:', 'M:', 'Q:', 'V:', 'I:']

    global_K = 'none'
    # 滤掉除 Q:K:M:L:V:I: 以外的 information field
    # 滤掉除 %%score 以外的 %%行
    V_found_flag = False
    filtered_abc_text_lines = []
    for i, line in enumerate(abc_text_lines):
        save_state = True
        if re.search(r'^[A-Za-z]:', line) and line[:2] not in reserved_info_field:
            save_state = False
        if line.startswith("%") and not line.startswith('%%score'):
            save_state = False
        if line.startswith('V:'):
            V_found_flag = True
        if line.startswith('K:') and not V_found_flag:
            global_K = line.lstrip('K:').strip()
        if save_state:
            filtered_abc_text_lines.append(line)

    # 分割为各个声部
    part_symbol_list = []
    part_symbol_Key_dict = {}

    tunebody_index = None
    for i, line in enumerate(filtered_abc_text_lines):
        if line.strip() == 'V:1':
            tunebody_index = i
            break
    if tunebody_index is None:
        raise Exception('tunebody index not found.')

    tunebody_lines = filtered_abc_text_lines[tunebody_index:]
    metadata_lines = filtered_abc_text_lines[:tunebody_index]
    part_text_list = []

    last_start_index = None
    for i, line in enumerate(tunebody_lines):
        if i == 0:
            last_start_index = 1
            part_symbol_list.append(line[:3])
            part_symbol_Key_dict[line[:3]] = 'none'
            continue
        if line.startswith('V:'):
            last_end_index = i
            part_text_list.append(
                ''.join(tunebody_lines[last_start_index:last_end_index]))
            part_symbol_list.append(line[:3])
            last_start_index = i + 1
    part_text_list.append(''.join(tunebody_lines[last_start_index:]))

    # 写入每个声部的初始Key
    for part_symbol in part_symbol_list:
        V_found_flag = False
        K_found_flag = False
        for j, line in enumerate(metadata_lines):
            if line.startswith(part_symbol):
                V_found_flag = True
            if line.startswith('K:'):
                K_found_flag = True
                part_symbol_Key_dict[part_symbol] = line.lstrip('K:').strip()
            if line.startswith('V:') and V_found_flag:
                break
        if not K_found_flag and global_K != 'none':
            part_symbol_Key_dict[part_symbol] = global_K

    # 如果无 global_K，则第一个有key的声部的key算global_key
    if global_K == 'none':
        for part_symbol in part_symbol_list:
            if part_symbol_Key_dict[part_symbol] != 'none':
                global_K = part_symbol_Key_dict[part_symbol]
                break

    transposed_abc_text = ''
    # 开始转调
    for line in metadata_lines:
        if line.startswith('K:'):
            key = line.lstrip('K:').strip()
            if key != 'none':
                transposed_key = lookup_new_key_to_transpose(
                    key, global_K, des_key)
            else:
                transposed_key = 'none'
            transposed_line = 'K:' + transposed_key + '\n'
        else:
            transposed_line = line
        transposed_abc_text += transposed_line

    for i, part_text in enumerate(part_text_list):
        transposed_abc_text += part_symbol_list[i] + '\n'
        part_symbol = part_symbol_list[i]
        part_ori_key = part_symbol_Key_dict[part_symbol]
        if part_ori_key == 'none':
            transposed_abc_text += part_text
        else:
            part_des_key = lookup_new_key_to_transpose(
                part_ori_key, global_K, des_key)
            transposed_part_text = transpose_a_voice(
                part_text, part_ori_key, part_des_key)
            transposed_abc_text += transposed_part_text

    return transposed_abc_text


def transpose_dataset():
    for abc_path in find_all_abc('04_abc_cleaned\\piano'):
        abc_name = abc_path.split('\\')[-1].split('.')[0]

        # if abc_name != '100701':
        #     continue

        print(abc_path)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_text_lines = f.readlines()

        print('X:1')
        print(''.join(abc_text_lines))
        print('')

        transposed_abc_text = transpose_an_abc_text(abc_text_lines, 'C')

        print('X:2')
        print(transposed_abc_text)
        print('')


if __name__ == '__main__':
    # transpose_dataset()

    # transpose_a_voice(test, 'C', 'C')
    # note = transpose_a_note(r"^^c''", 'C', 'Eb')
    # print(note)

    with open('./data/test.abc', 'r', encoding='utf-8') as f:
        abc_text_lines = f.readlines()

    print('X:1')
    print(''.join(abc_text_lines))
    print('')

    transposed_abc_text = transpose_an_abc_text(abc_text_lines, 'A')

    print('X:2')
    print(transposed_abc_text)
    print('')
