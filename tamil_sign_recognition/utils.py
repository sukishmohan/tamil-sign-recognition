# utils.py
# Utility functions and label mappings for Tamil Sign Recognition
# Folder numbers 1-247 = Tamil characters, 248 = Background

import os
import sys

# ── Complete label map: folder number → Tamil character ──────────────────────
LABEL_MAP = {
    1:   'அ',   2:   'ஆ',   3:   'இ',   4:   'ஈ',   5:   'உ',
    6:   'ஊ',   7:   'எ',   8:   'ஏ',   9:   'ஐ',   10:  'ஒ',
    11:  'ஓ',   12:  'ஔ',   13:  'ஃ',   14:  'க்',  15:  'ங்',
    16:  'ச்',  17:  'ஞ்',  18:  'ட்',  19:  'ண்',  20:  'த்',
    21:  'ந்',  22:  'ப்',  23:  'ம்',  24:  'ய்',  25:  'ர்',
    26:  'ல்',  27:  'வ்',  28:  'ழ்',  29:  'ள்',  30:  'ற்',
    31:  'ன்',  32:  'க',   33:  'கா',  34:  'கி',  35:  'கீ',
    36:  'கு',  37:  'கூ',  38:  'கெ',  39:  'கே',  40:  'கை',
    41:  'கொ',  42:  'கோ',  43:  'கௌ',  44:  'ங',   45:  'ஙா',
    46:  'ஙி',  47:  'ஙீ',  48:  'ஙு',  49:  'ஙூ',  50:  'ஙெ',
    51:  'ஙே',  52:  'ஙை',  53:  'ஙொ',  54:  'ஙோ',  55:  'ஙௌ',
    56:  'ச',   57:  'சா',  58:  'சி',  59:  'சீ',  60:  'சு',
    61:  'சூ',  62:  'செ',  63:  'சே',  64:  'சை',  65:  'சொ',
    66:  'சோ',  67:  'சௌ',  68:  'ஞ',   69:  'ஞா',  70:  'ஞி',
    71:  'ஞீ',  72:  'ஞு',  73:  'ஞூ',  74:  'ஞெ',  75:  'ஞே',
    76:  'ஞை',  77:  'ஞொ',  78:  'ஞோ',  79:  'ஞௌ',  80:  'ட',
    81:  'டா',  82:  'டி',  83:  'டீ',  84:  'டு',  85:  'டூ',
    86:  'டெ',  87:  'டே',  88:  'டை',  89:  'டொ',  90:  'டோ',
    91:  'டௌ',  92:  'ண',   93:  'ணா',  94:  'ணி',  95:  'ணீ',
    96:  'ணு',  97:  'ணூ',  98:  'ணெ',  99:  'ணே',  100: 'ணை',
    101: 'ணொ',  102: 'ணோ',  103: 'ணௌ',  104: 'த',   105: 'தா',
    106: 'தி',  107: 'தீ',  108: 'து',  109: 'தூ',  110: 'தெ',
    111: 'தே',  112: 'தை',  113: 'தொ',  114: 'தோ',  115: 'தௌ',
    116: 'ந',   117: 'நா',  118: 'நி',  119: 'நீ',  120: 'நு',
    121: 'நூ',  122: 'நெ',  123: 'நே',  124: 'நை',  125: 'நொ',
    126: 'நோ',  127: 'நௌ',  128: 'ப',   129: 'பா',  130: 'பி',
    131: 'பீ',  132: 'பு',  133: 'பூ',  134: 'பெ',  135: 'பே',
    136: 'பை',  137: 'பொ',  138: 'போ',  139: 'பௌ',  140: 'ம',
    141: 'மா',  142: 'மி',  143: 'மீ',  144: 'மு',  145: 'மூ',
    146: 'மெ',  147: 'மே',  148: 'மை',  149: 'மொ',  150: 'மோ',
    151: 'மௌ',  152: 'ய',   153: 'யா',  154: 'யி',  155: 'யீ',
    156: 'யு',  157: 'யூ',  158: 'யெ',  159: 'யே',  160: 'யை',
    161: 'யொ',  162: 'யோ',  163: 'யௌ',  164: 'ர',   165: 'ரா',
    166: 'ரி',  167: 'ரீ',  168: 'ரு',  169: 'ரூ',  170: 'ரெ',
    171: 'ரே',  172: 'ரை',  173: 'ரொ',  174: 'ரோ',  175: 'ரௌ',
    176: 'ல',   177: 'லா',  178: 'லி',  179: 'லீ',  180: 'லு',
    181: 'லூ',  182: 'லெ',  183: 'லே',  184: 'லை',  185: 'லொ',
    186: 'லோ',  187: 'லௌ',  188: 'வ',   189: 'வா',  190: 'வி',
    191: 'வீ',  192: 'வு',  193: 'வூ',  194: 'வெ',  195: 'வே',
    196: 'வை',  197: 'வொ',  198: 'வோ',  199: 'வௌ',  200: 'ழ',
    201: 'ழா',  202: 'ழி',  203: 'ழீ',  204: 'ழு',  205: 'ழூ',
    206: 'ழெ',  207: 'ழே',  208: 'ழை',  209: 'ழொ',  210: 'ழோ',
    211: 'ழௌ',  212: 'ள',   213: 'ளா',  214: 'ளி',  215: 'ளீ',
    216: 'ளு',  217: 'ளூ',  218: 'ளெ',  219: 'ளே',  220: 'ளை',
    221: 'ளொ',  222: 'ளோ',  223: 'ளௌ',  224: 'ற',   225: 'றா',
    226: 'றி',  227: 'றீ',  228: 'று',  229: 'றூ',  230: 'றெ',
    231: 'றே',  232: 'றை',  233: 'றொ',  234: 'றோ',  235: 'றௌ',
    236: 'ன',   237: 'னா',  238: 'னி',  239: 'னீ',  240: 'னு',
    241: 'னூ',  242: 'னெ',  243: 'னே',  244: 'னை',  245: 'னொ',
    246: 'னோ',  247: 'னௌ',  248: 'Background',
}

# Number of classes
NUM_CLASSES = 248

# Background class index (0-based, folder 248 → index 247)
BACKGROUND_CLASS_IDX = 247

# Sorted folder names as strings (the way ImageDataGenerator sees them)
# Numeric folders: "1"–"247"; background folder is literally named "Background"
CLASS_FOLDERS = [str(i) for i in range(1, NUM_CLASSES)] + ['Background']

# Map: 0-based class index → Tamil character
# ImageDataGenerator sorts folder names alphabetically (string sort),
# so we must reproduce that ordering here.
def build_index_to_label():
    """
    Keras ImageDataGenerator sorts class folder names alphabetically.
    Numeric folder names ("1","2",...,"248") sort as strings:
    1, 10, 100, ..., 109, 11, 110, ..., etc.
    This function reconstructs that exact ordering.
    """
    sorted_folders = sorted(CLASS_FOLDERS, key=lambda x: x)  # string sort
    # Note: 'Background' > any digit-starting string, so it lands last (index 247)
    idx_to_label = {}
    idx_to_char = {}
    for idx, folder in enumerate(sorted_folders):
        # Map the string folder name to its logical number in LABEL_MAP
        folder_num = 248 if folder == 'Background' else int(folder)
        char = LABEL_MAP[folder_num]
        idx_to_label[idx] = folder_num
        idx_to_char[idx] = char
    return idx_to_label, idx_to_char, sorted_folders


IDX_TO_FOLDER, IDX_TO_CHAR, SORTED_FOLDERS = build_index_to_label()

# Reverse: Tamil char → list of class indices (multiple indices may share a char)
CHAR_TO_IDX = {}
for idx, char in IDX_TO_CHAR.items():
    CHAR_TO_IDX.setdefault(char, []).append(idx)


def get_tamil_char(class_index: int) -> str:
    """Return the Tamil character for a given 0-based class index."""
    return IDX_TO_CHAR.get(class_index, '?')


def get_folder_number(class_index: int) -> int:
    """Return the dataset folder number for a given 0-based class index."""
    return IDX_TO_FOLDER.get(class_index, -1)


def is_background(class_index: int) -> bool:
    """Return True if the prediction is the background class (folder 248)."""
    return IDX_TO_FOLDER.get(class_index, -1) == 248


BACKGROUND_FOLDER_NAME = 'Background'  # Actual folder name in the dataset


def setup_utf8_console():
    """Force UTF-8 output on Windows so Tamil characters print correctly."""
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace'
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding='utf-8', errors='replace'
        )
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass


def load_word_list(readme_path: str) -> list:
    """
    Parse the README.txt and extract Tamil words listed in it.
    Returns a list of Tamil word strings.
    """
    words = []
    if not os.path.exists(readme_path):
        return words
    try:
        with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # Collect non-empty lines that contain Tamil Unicode characters
                if line and any('\u0B80' <= ch <= '\u0BFF' for ch in line):
                    words.append(line)
    except Exception:
        pass
    return words
