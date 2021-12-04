import platform
from copy import deepcopy

import torch
import os
import logging
from glob import glob
from typing import Union, List, Tuple, Optional, Collection, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import hashlib
import zipfile
import requests
from torch import Tensor, nn
from tqdm import tqdm

from  itertools import groupby
from torchvision.models import densenet
from typing import Tuple, Dict, Any, Optional, List, Union
from copy import deepcopy

import numpy as np
import torch
from torch import nn

import torch.nn.functional as Fun
import torchvision.transforms.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import cv2
from cnstd import CnStd






VOCAB_FP = Path(__file__).parent / 'label_cn.txt'

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024.0 + 0.5),
                    unit='KB',
                    unit_scale=False,
                    dynamic_ncols=True,
                ):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                'File {} is downloaded but the content hash does not match. '
                'The repo may be outdated or download may be incomplete. '
                'If the "repo_url" is overridden, consider switching to '
                'the default repo.'.format(fname)
            )
    return fname

def get_model_file(model_dir):
    r"""Return location for the downloaded models on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_dir : str, default $CNOCR_HOME
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    model_dir = os.path.expanduser(model_dir)
    par_dir = os.path.dirname(model_dir)
    os.makedirs(par_dir, exist_ok=True)

    zip_file_path = model_dir + '.zip'
    if not os.path.exists(zip_file_path):
        model_name = os.path.basename(model_dir)
        if model_name not in AVAILABLE_MODELS:
            raise NotImplementedError(
                '%s is not an available downloaded model' % model_name
            )
        url = AVAILABLE_MODELS[model_name][1]
        download(url, path=zip_file_path, overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(par_dir)
    os.remove(zip_file_path)

    return model_dir

def pad_img_seq(img_list: List[torch.Tensor], padding_value=0) -> torch.Tensor:
    """
    Pad a list of variable width image Tensors with `padding_value`.

    :param img_list: each element has shape [C, H, W], where W is variable width
    :param padding_value: padding value, 0 by default
    :return: [B, C, H, W_max]
    """
    img_list = [img.permute((2, 0, 1)) for img in img_list]  # [W, C, H]
    imgs = pad_sequence(
        img_list, batch_first=True, padding_value=padding_value
    )  # [B, W_max, C, H]
    return imgs.permute((0, 2, 3, 1))  # [B, C, H, W_max]

def _assert_and_prepare_model_files(self, model_fp, root):
    if model_fp is not None and not os.path.isfile(model_fp):
        raise FileNotFoundError('can not find model file %s' % model_fp)

    if model_fp is not None:
        self._model_fp = model_fp
        return

    root = os.path.join(root, MODEL_VERSION)
    self._model_dir = os.path.join(root, self._model_name)
    fps = glob('%s/%s*.ckpt' % (self._model_dir, self._model_file_prefix))
    if len(fps) > 1:
        raise ValueError(
            'multiple ckpt files are found in %s, not sure which one should be used'
            % self._model_dir
        )
    elif len(fps) < 1:
        get_model_file(self._model_dir)  # download the .zip file and unzip
        fps = glob('%s/%s*.ckpt' % (self._model_dir, self._model_file_prefix))

    self._model_fp = fps[0]


def check_model_name(model_name):
    encoder_type, decoder_type = model_name.rsplit('-', maxsplit=1)
    assert encoder_type in ENCODER_CONFIGS
    assert decoder_type in DECODER_CONFIGS
def check_context(context):
    if isinstance(context, str):
        return any([ctx in context.lower() for ctx in ('gpu', 'cpu', 'cuda')])
    if isinstance(context, list):
        if len(context) < 1:
            return False
        return all(isinstance(ctx, torch.device) for ctx in context)
    return isinstance(context, torch.device)

__version__ = '2.0.1'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
root_url = (
    'https://beiye-model.oss-cn-beijing.aliyuncs.com/models/cnocr/%s/'
    % MODEL_VERSION
)
AVAILABLE_MODELS = {
    'densenet-s-fc': (8, root_url + 'densenet-s-fc-v2.0.1.zip'),
    'densenet-s-gru': (14, root_url + 'densenet-s-gru-v2.0.1.zip'),
}
IMG_STANDARD_HEIGHT = 32

def rescale_img(img: np.ndarray) -> torch.Tensor:
    """
    rescale an image tensor with [Channel, Height, Width] to the given height value, and keep the ratio
    :param img: np.ndarray; should be [c, height, width]
    :return: image tensor with the given height. The resulting dim is [C, height, width]
    """
    ori_height, ori_width = img.shape[1:]
    ratio = ori_height / IMG_STANDARD_HEIGHT
    img = torch.from_numpy(img)
    if img.size(1) != IMG_STANDARD_HEIGHT:
        img = F.resize(img, [IMG_STANDARD_HEIGHT, int(ori_width / ratio)])
    return img


def read_charset(charset_fp):
    alphabet = []
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    if len(alphabet) != len(inv_alph_dict):
        from collections import Counter

        repeated = Counter(alphabet).most_common(len(alphabet) - len(inv_alph_dict))
        raise ValueError('repeated chars in vocab: %s' % repeated)

    return alphabet, inv_alph_dict

def set_cand_alphabet(self, cand_alphabet: Optional[Union[Collection, str]]):
    """
    设置待识别字符的候选集合。
    :param cand_alphabet: 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围
    :return: None
    """
    if cand_alphabet is None:
        self._candidates = None
    else:
        cand_alphabet = [
            word if word != ' ' else '<space>' for word in cand_alphabet
        ]
        excluded = set(
            [word for word in cand_alphabet if word not in self._letter2id]
        )

        candidates = [word for word in cand_alphabet if word in self._letter2id]
        self._candidates = None if len(candidates) == 0 else candidates

def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'cnocr')
    else:
        return os.path.join(os.path.expanduser("~"), '.cnocr')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('CNOCR_HOME', data_dir_default())

def read_img(path: Union[str, Path], gray=True) -> np.ndarray:
    """
    :param path: image file path
    :param gray: whether to return a gray image array
    :return:
        * when `gray==True`, return a gray image, with dim [height, width, 1], with values range from 0 to 255
        * when `gray==False`, return a color image, with dim [height, width, 3], with values range from 0 to 255
    """
    img = Image.open(path)
    if gray:
        return np.expand_dims(np.array(img.convert('L')), -1)
    else:
        return np.asarray(img.convert('RGB'))


THRESHOLD = 145  # for white background
TABLE = [1]*THRESHOLD + [0]*(256-THRESHOLD)
ENCODER_CONFIGS = {
    'densenet-s': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 4*128 = 512
        'growth_rate': 32,
        'block_config': [2, 2, 2, 2],
        'num_init_features': 64,
        'out_length': 512,  # 输出的向量长度为 4*128 = 512
    },
}
DECODER_CONFIGS = {
    'lstm': {
        'input_size': 512,  # 对应 encoder 的输出向量长度
        'rnn_units': 128,
    },
    'gru': {
        'input_size': 512,  # 对应 encoder 的输出向量长度
        'rnn_units': 128,
    },
    'fc': {
        'input_size': 512,  # 对应 encoder 的输出向量长度
        'hidden_size': 256,
        'dropout': 0.3,
    }
}
def line_split(image, table=TABLE, split_threshold=0, blank=True):
    """
    :param image: PIL.Image类型的原图或numpy.ndarray
    :param table: 二值化的分布值，默认值即可
    :param split_threshold: int, 分割阈值
    :param blank: bool,是否留白.True会保留上下方的空白部分
    :return: list,元素为按行切分出的子图与位置信息的list
    """
    if not isinstance(image, Image.Image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise TypeError

    image_ = image.convert('L')
    bn = image_.point(table, '1')
    bn_mat = np.array(bn)
    h, pic_len = bn_mat.shape
    project = np.sum(bn_mat, 1)
    pos = np.where(project <= split_threshold)[0]
    if len(pos) == 0 or pos[0] != 0:
        pos = np.insert(pos, 0, 0)
    if pos[-1] != len(project):
        pos = np.append(pos, len(project))
    diff = np.diff(pos)

    if len(diff) == 0:
        return [[np.array(image), (0, 0, pic_len, h)]]

    width = np.max(diff)
    coordinate = list(zip(pos[:-1], pos[1:]))
    info = list(zip(diff, coordinate))
    info = list(filter(lambda x: x[0] > 10, info))

    split_pos = []
    temp = []
    for pos_info in info:
        if width-2 <= pos_info[0] <= width:
            if temp:
                split_pos.append(temp.pop(0))
            split_pos.append(pos_info)

        elif pos_info[0] < width-2:
            temp.append(pos_info)
            if len(temp) > 1:
                s, e = temp[0][1][0], temp[1][1][1]
                if e - s <= width + 2:
                    temp = [(e - s, (s, e))]
                else:
                    split_pos.append(temp.pop(0))

    if temp:
        split_pos.append(temp[0])

    # crop images with split_pos
    line_res = []
    if blank:
        if len(split_pos) == 1:
            pos_info = split_pos[0][1]
            ymin, ymax = max(0, pos_info[0]-2), min(h, pos_info[1]+2)
            return [[np.array(image.crop((0, ymin, pic_len, ymax))), (0, ymin, pic_len, ymax)]]

        length = len(split_pos)
        for i in range(length):
            if i == 0:
                next_info = split_pos[i+1]
                margin = min(next_info[1][0] - split_pos[i][1][1], 2)
                ymin, ymax = max(0, split_pos[i][1][0] - margin), split_pos[i][1][1] + margin
                x1, y1, x2, y2 = 0, ymin, pic_len, ymax
                sub = image.crop((x1, y1, x2, y2))
            elif i == length-1:
                pre_info = split_pos[i - 1]
                margin = min(split_pos[i][1][0] - pre_info[1][1], 2)
                ymin, ymax = split_pos[i][1][0] - margin, min(h, split_pos[i][1][1] + margin)
                x1, y1, x2, y2 = 0, ymin, pic_len, ymax
                sub = image.crop((x1, y1, x2, y2))
            else:
                next_info = split_pos[i + 1]
                pre_info = split_pos[i - 1]
                margin = min(split_pos[i][1][0] - pre_info[1][1], next_info[1][0] - split_pos[i][1][0], 2)
                ymin, ymax = split_pos[i][1][0] - margin, split_pos[i][1][1] + margin
                x1, y1, x2, y2 = 0, ymin, pic_len, ymax
                sub = image.crop((x1, y1, x2, y2))

            line_res.append([np.array(sub), (x1, y1, x2, y2)])
    else:
        for pos_info in split_pos:
            x1, y1, x2, y2 = 0, pos_info[1][0], pic_len, pos_info[1][1]
            sub = image.crop((x1, y1, x2, y2))
            line_res.append([np.array(sub), (x1, y1, x2, y2)])

    return line_res

def normalize_img_array(img: Union[Tensor, np.ndarray]):
    """ rescale """
    if isinstance(img, Tensor):
        img = img.to(dtype=torch.float32)
    else:
        img = img.astype('float32')
    # return (img - np.mean(img, dtype=dtype)) / 255.0
    return img / 255.0
    # return (img - np.median(img)) / (np.std(img, dtype=dtype) + 1e-6)  # 转完以后有些情况会变得不可识别

def gen_model(model_name, vocab):
    check_model_name(model_name)
    if not model_name.startswith('densenet-s'):
        model_name = 'densenet-s-fc'
    model = OcrModel.from_name(model_name, vocab)
    return model
def gen_length_mask(lengths: torch.Tensor, mask_size: Union[Tuple, Any]):
    """ see how it is used """
    labels = torch.arange(mask_size[-1], device=lengths.device, dtype=torch.long)
    while True:
        if len(labels.shape) >= len(mask_size):
            break
        labels = labels.unsqueeze(0)
        lengths = lengths.unsqueeze(-1)
    mask = labels < lengths
    return ~mask

def encode_sequence(input_string: str, vocab: Dict[str, int],) -> List[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
        input_string: string to encode
        vocab: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A list encoding the input_string"""

    return [vocab[letter] for letter in input_string]
    # return list(map(vocab.index, input_string))  # type: ignore[arg-type]


def decode_sequence(input_array: np.array, mapping: str,) -> str:
    """Given a predefined mapping, decode the sequence of numbers to a string

    Args:
        input_array: array to decode
        mapping: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A string, decoded from input_array"""

    if not input_array.dtype == np.int_ or input_array.max() >= len(mapping):
        raise AssertionError(
            "Input must be an array of int, with max less than mapping size"
        )
    decoded = ''.join(mapping[idx] for idx in input_array)
    return decoded


def encode_sequences(
    sequences: List[str],
    vocab: Dict[str, int],
    target_size: Optional[int] = None,
    eos: int = -1,
    sos: Optional[int] = None,
    pad: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Encode character sequences using a given vocab as mapping

    Args:
        sequences: the list of character sequences of size N
        vocab: the ordered vocab to use for encoding
        target_size: maximum length of the encoded data
        eos: encoding of End Of String
        sos: optional encoding of Start Of String
        pad: optional encoding for padding. In case of padding, all sequences are followed by 1 EOS then PAD

    Returns:
        the padded encoded data as a tensor
    """

    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")

    if not isinstance(target_size, int):
        target_size = max(len(w) for w in sequences)
        if sos:
            target_size += 1
        if pad:
            target_size += 1

    # Pad all sequences
    if pad:  # pad with padding symbol
        if 0 <= pad < len(vocab):
            raise ValueError(
                "argument 'pad' needs to be outside of vocab possible indices"
            )
        # In that case, add EOS at the end of the word before padding
        encoded_data = np.full([len(sequences), target_size], pad, dtype=np.int32)
    else:  # pad with eos symbol
        encoded_data = np.full([len(sequences), target_size], eos, dtype=np.int32)

    for idx, seq in enumerate(sequences):
        encoded_seq = encode_sequence(seq, vocab)
        if pad:  # add eos at the end of the sequence
            encoded_seq.append(eos)
        encoded_data[idx, : min(len(encoded_seq), target_size)] = encoded_seq[
            : min(len(encoded_seq), target_size)
        ]

    if sos:  # place eos symbol at the beginning of each sequence
        if 0 <= sos < len(vocab):
            raise ValueError(
                "argument 'sos' needs to be outside of vocab possible indices"
            )
        encoded_data = np.roll(encoded_data, 1)
        encoded_data[:, 0] = sos

    return encoded_data

class NormalizeAug(object):
    def __call__(self, img):
        return normalize_img_array(img)


class CTCPostProcessor(object):
    """
    Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: List[str],) -> None:

        self.vocab = vocab
        self._embedding = list(self.vocab) + ['<eos>']

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: List[str],
        input_lengths: Optional[torch.Tensor] = None,
        blank: int = 0,
    ) -> List[Tuple[List[str], float]]:
        """Implements best path decoding as shown by Graves (Dissertation, p63), highly inspired from
        <https://github.com/githubharald/CTCDecoder>`_.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            input_lengths: valid sequence lengths
            blank: index of blank label

        Returns:
            A list of tuples: (word, confidence)
        """
        # compute softmax
        probs = Fun.softmax(logits.permute(0, 2, 1), dim=1)
        # get char indices along best path
        best_path = torch.argmax(probs, dim=1)  # [N, T]

        if input_lengths is not None:
            length_mask = gen_length_mask(input_lengths, probs.shape).to(
                device=probs.device
            )  # [N, 1, T]
            probs.masked_fill_(length_mask, 1.0)
            best_path.masked_fill_(length_mask.squeeze(1), blank)

        # define word proba as min proba of sequence
        probs, _ = torch.max(probs, dim=1)  # [N, T]
        probs, _ = torch.min(probs, dim=1)  # [N]

        words = []
        for sequence in best_path:
            # collapse best path (using itertools.groupby), map to chars, join char list to string
            collapsed = [vocab[k] for k, _ in groupby(sequence) if k != blank]
            words.append(collapsed)

        return list(zip(words, probs.tolist()))

    def __call__(  # type: ignore[override]
        self, logits: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> List[Tuple[List[str], float]]:
        """
        Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)
            input_lengths: valid sequence lengths

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Decode CTC
        return self.ctc_best_path(
            logits=logits,
            vocab=self.vocab,
            input_lengths=input_lengths,
            blank=len(self.vocab),
        )

class DenseNet(densenet.DenseNet):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__(
            growth_rate,
            block_config,
            num_init_features,
            bn_size,
            drop_rate,
            num_classes=1,  # useless, will be deleted
            memory_efficient=memory_efficient,
        )

        self.block_config = block_config
        self.features.conv0 = nn.Conv2d(
            1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.features.pool0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        delattr(self, 'classifier')
        self._post_init_weights()

    @property
    def compress_ratio(self):
        return 2 ** (len(self.block_config) - 1)

    def _post_init_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        return features


class EncoderManager(object):
    @classmethod
    def gen_encoder(
        cls, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> Tuple[nn.Module, int]:
        if name is not None:
            assert name in ENCODER_CONFIGS
            config = deepcopy(ENCODER_CONFIGS[name])
        else:
            assert config is not None and 'name' in config
            name = config.pop('name')

        if name.lower() == 'densenet-s':
            out_length = config.pop('out_length')
            encoder = DenseNet(**config)
        else:
            raise ValueError('not supported encoder name: %s' % name)
        return encoder, out_length

class DecoderManager(object):
    @classmethod
    def gen_decoder(
        cls,
        input_size: int,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, int]:
        if name is not None:
            assert name in DECODER_CONFIGS
            config = deepcopy(DECODER_CONFIGS[name])
        else:
            assert config is not None and 'name' in config
            name = config.pop('name')

        if name.lower() == 'lstm':
            decoder = nn.LSTM(
                input_size=input_size,
                hidden_size=config['rnn_units'],
                batch_first=True,
                num_layers=2,
                bidirectional=True,
            )
            out_length = config['rnn_units'] * 2
        elif name.lower() == 'gru':
            decoder = nn.GRU(
                input_size=input_size,
                hidden_size=config['rnn_units'],
                batch_first=True,
                num_layers=2,
                bidirectional=True,
            )
            out_length = config['rnn_units'] * 2
        elif name.lower() == 'fc':
            decoder = nn.Sequential(
                nn.Dropout(p=config['dropout']),
                # nn.Tanh(),
                nn.Linear(config['input_size'], config['hidden_size']),
                nn.Dropout(p=config['dropout']),
                nn.Tanh(),
            )
            out_length = config['hidden_size']
        else:
            raise ValueError('not supported encoder name: %s' % name)
        return decoder, out_length

class OcrModel(nn.Module):
    """OCR Model.

    Args:
        encoder: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        cfg: configuration dictionary
    """

    _children_names: List[str] = [
        'encoder',
        'decoder',
        'linear',
        'postprocessor',
    ]

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        decoder_out_length: int,
        vocab: List[str],
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.letter2id = {letter: idx for idx, letter in enumerate(self.vocab)}
        assert len(self.vocab) == len(self.letter2id)

        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(decoder_out_length, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    @classmethod
    def from_name(cls, name: str, vocab: List[str]):
        encoder_name, decoder_name = name.rsplit('-', maxsplit=1)
        encoder, encoder_out_len = EncoderManager.gen_encoder(encoder_name)
        decoder, decoder_out_len = DecoderManager.gen_decoder(
            encoder_out_len, decoder_name
        )
        return cls(encoder, decoder, decoder_out_len, vocab)

    def calculate_loss(
        self, batch, return_model_output: bool = False, return_preds: bool = False,
    ):
        imgs, img_lengths, labels_list, label_lengths = batch
        return self(
            imgs, img_lengths, labels_list, None, return_model_output, return_preds
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        target: Optional[List[str]] = None,
        candidates: Optional[Union[str, List[str]]] = None,
        return_logits: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """

        :param x: [B, 1, H, W]; 一组padding后的图片
        :param input_lengths: shape: [B]；每张图片padding前的真实长度（宽度）
        :param target: 真实的字符串
        :param candidates: None or candidate strs; 允许的候选字符集合
        :param return_logits: 是否返回预测的logits值
        :param return_preds: 是否返回预测的字符串
        :return: 预测结果
        """
        features = self.encoder(x)
        input_lengths = input_lengths // self.encoder.compress_ratio
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)  # B x W x C*H

        logits = self._decode(features_seq, input_lengths)

        logits = self.linear(logits)
        logits = self._mask_by_candidates(logits, candidates)

        out: Dict[str, Any] = {}
        if return_logits:
            out["logits"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits, input_lengths)

        if target is not None:
            out['loss'] = self._compute_loss(logits, target, input_lengths)

        return out

    def _decode(self, features_seq, input_lengths):
        if not isinstance(self.decoder, (nn.LSTM, nn.GRU)):
            return self.decoder(features_seq)

        w = features_seq.shape[1]
        features_seq = pack_padded_sequence(
            features_seq,
            input_lengths.to(device='cpu'),
            batch_first=True,
            enforce_sorted=False,
        )
        logits, _ = self.decoder(features_seq)
        logits, output_lens = pad_packed_sequence(
            logits, batch_first=True, total_length=w
        )
        return logits

    def _mask_by_candidates(
        self, logits: torch.Tensor, candidates: Optional[Union[str, List[str]]]
    ):
        if candidates is None:
            return logits

        _candidates = [self.letter2id[word] for word in candidates]
        _candidates.sort()
        _candidates = torch.tensor(_candidates, dtype=torch.int64)

        candidates = torch.zeros(
            (len(self.vocab) + 1,), dtype=torch.bool, device=logits.device
        )
        candidates[_candidates] = True
        candidates[-1] = True  # 间隔符号/填充符号，必须为真
        candidates = candidates.unsqueeze(0).unsqueeze(0)  # 1 x 1 x (vocab_size+1)
        logits.masked_fill_(~candidates, -100.0)
        return logits

    def _compute_loss(
        self,
        model_output: torch.Tensor,
        target: List[str],
        seq_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_length: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.compute_target(target)

        if seq_length is None:
            batch_len = model_output.shape[0]
            seq_length = model_output.shape[1] * torch.ones(
                size=(batch_len,), dtype=torch.int32
            )

        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)

        ctc_loss = F.ctc_loss(
            probs,
            torch.from_numpy(gt).to(device=probs.device),
            seq_length,
            torch.tensor(seq_len, dtype=torch.int, device=probs.device),
            len(self.vocab),
            zero_infinity=True,
        )

        return ctc_loss

    def compute_target(self, gts: List[str],) -> Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(
            sequences=gts, vocab=self.letter2id, eos=len(self.letter2id),
        )
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class CnOcr(object):
    MODEL_FILE_PREFIX = 'cnocr-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        model_name: str = 'densenet-s-fc',
        *,
        cand_alphabet: Optional[Union[Collection, str]] = None,
        context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        **kwargs,
    ):
        check_model_name(model_name)
        check_context(context)
        self._model_name = model_name
        if context == 'gpu':
            context = 'cuda'
        self.context = context
        # format prefix of a model file
        self._model_file_prefix = '{}-{}'.format(self.MODEL_FILE_PREFIX, model_name)

        model_epoch = AVAILABLE_MODELS.get(model_name, [None])[0]
        if model_epoch is not None:
            self._model_file_prefix = '%s-epoch=%03d' % (
                self._model_file_prefix,
                model_epoch,
            )
        self._assert_and_prepare_model_files(model_fp, root)
        self._vocab, self._letter2id = read_charset(VOCAB_FP)

        self._candidates = None
        self.set_cand_alphabet(cand_alphabet)
        self._model = self._get_model(context)

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, self._model_name)
        fps = glob('%s/%s*.ckpt' % (self._model_dir, self._model_file_prefix))
        if len(fps) > 1:
            raise ValueError(
                'multiple ckpt files are found in %s, not sure which one should be used'
                % self._model_dir
            )
        elif len(fps) < 1:
            get_model_file(self._model_dir)  # download the .zip file and unzip
            fps = glob('%s/%s*.ckpt' % (self._model_dir, self._model_file_prefix))

        self._model_fp = fps[0]

    def _get_model(self, context):

        model = gen_model(self._model_name, self._vocab)
        model.eval()
        model.to(self.context)
        model = load_model_params(model, self._model_fp, context)

        return model

    def set_cand_alphabet(self, cand_alphabet: Optional[Union[Collection, str]]):
        """
        设置待识别字符的候选集合。
        :param cand_alphabet: 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围
        :return: None
        """
        if cand_alphabet is None:
            self._candidates = None
        else:
            cand_alphabet = [
                word if word != ' ' else '<space>' for word in cand_alphabet
            ]
            excluded = set(
                [word for word in cand_alphabet if word not in self._letter2id]
            )

            candidates = [word for word in cand_alphabet if word in self._letter2id]
            self._candidates = None if len(candidates) == 0 else candidates

    def ocr(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> List[Tuple[List[str], float]]:
        """
        :param img_fp: image file path; or color image torch.Tensor or np.ndarray,
            with shape [height, width] or [height, width, channel].
            channel should be 1 (gray image) or 3 (RGB formatted color image). scaled in [0, 255].
        :return: list of (list of chars, prob), such as
            [(['第', '一', '行'], 0.80), (['第', '二', '行'], 0.75), (['第', '三', '行'], 0.9)]
        """
        img = self._prepare_img(img_fp)

        if min(img.shape[0], img.shape[1]) < 2:
            return []
        if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
            img = 255 - img
        line_imgs = line_split(np.squeeze(img, axis=-1), blank=True)
        line_img_list = [np.expand_dims(line_img, axis=-1) for line_img, _ in line_imgs]
        line_chars_list = self.ocr_for_single_lines(line_img_list)
        return line_chars_list

    def _prepare_img(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        :param img: image array with type torch.Tensor or np.ndarray,
        with shape [height, width] or [height, width, channel].
        channel should be 1 (gray image) or 3 (color image).

        :return: np.ndarray, with shape (height, width, 1), dtype uint8, scale [0, 255]
        """
        img = img_fp
        if isinstance(img_fp, (str, Path)):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                # color to gray
                img = np.expand_dims(np.array(Image.fromarray(img).convert('L')), -1)
            elif img.shape[2] != 1:
                raise ValueError(
                    'only images with shape [height, width, 1] (gray images), '
                    'or [height, width, 3] (RGB-formated color images) are supported'
                )

        if img.dtype != np.dtype('uint8'):
            img = img.astype('uint8')
        return img

    def ocr_for_single_line(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> Tuple[List[str], float]:
        """
        Recognize characters from an image with only one-line characters.
        :param img_fp: image file path; or image torch.Tensor or np.ndarray,
            with shape [height, width] or [height, width, channel].
            The optional channel should be 1 (gray image) or 3 (color image).
        :return: (list of chars, prob), such as (['你', '好'], 0.80)
        """
        img = self._prepare_img(img_fp)
        res = self.ocr_for_single_lines([img])
        return res[0]

    def ocr_for_single_lines(
        self,
        img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
        batch_size: int = 1,
    ) -> List[Tuple[List[str], float]]:
        """
        Batch recognize characters from a list of one-line-characters images.
        :param img_list: list of images, in which each element should be a line image array,
            with type torch.Tensor or np.ndarray.
            Each element should be a tensor with values ranging from 0 to 255,
            and with shape [height, width] or [height, width, channel].
            The optional channel should be 1 (gray image) or 3 (color image).
        :param batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。
        :return: list of (list of chars, prob), such as
            [(['第', '一', '行'], 0.80), (['第', '二', '行'], 0.75), (['第', '三', '行'], 0.9)]
        """
        if len(img_list) == 0:
            return []
        img_list = [self._prepare_img(img) for img in img_list]
        img_list = [self._transform_img(img) for img in img_list]

        idx = 0
        out = []
        while idx * batch_size < len(img_list):
            imgs = img_list[idx * batch_size : (idx + 1) * batch_size]
            batch_out = self._predict(imgs)
            out.extend(batch_out['preds'])
            idx += 1

        res = []
        for line in out:
            chars, prob = line
            chars = [c if c != '<space>' else ' ' for c in chars]
            res.append((chars, prob))

        return res


    def _transform_img(self, img: np.ndarray) -> torch.Tensor:
        """
        :param img: image array with type torch.Tensor or np.ndarray,
        with shape [height, width] or [height, width, channel].
        channel shoule be 1 (gray image) or 3 (color image).

        :return: torch.Tensor, with shape (1, height, width)
        """
        img = rescale_img(img.transpose((2, 0, 1)))  # res: [C, H, W]
        return NormalizeAug()(img).to(device=torch.device(self.context))

    @torch.no_grad()
    def _predict(self, img_list: List[torch.Tensor]):
        img_lengths = torch.tensor([img.shape[2] for img in img_list])
        imgs = pad_img_seq(img_list)
        out = self._model(
            imgs, img_lengths, candidates=self._candidates, return_preds=True
        )
        return out




def load_model_params(model, param_fp, device='cpu'):
    checkpoint = torch.load(param_fp, map_location=device)
    state_dict = checkpoint['state_dict']
    if all([param_name.startswith('model.') for param_name in state_dict.keys()]):
        # 表示导入的模型是通过 PlTrainer 训练出的 WrapperLightningModule，对其进行转化
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.split('.', maxsplit=1)[1]] = v
    model.load_state_dict(state_dict)
    return model

def _get_model(self, context):
    model = gen_model(self._model_name, self._vocab)
    model.eval()
    model.to(self.context)
    model = load_model_params(model, self._model_fp, context)

def main():
    img_path = r"E:\Collected\[Courses]\(3-1)机器视觉\Teamwork\daima\R-C (3).jfif"
    ocr = CnOcr()
    std = CnStd()

    box_infos = std.detect(img_path)
    for box_info in box_infos['detected_texts']:
        cropped_img = box_info['cropped_img']
        cv2.imshow("c", cropped_img)
        cv2.waitKey(0)
        result = ocr.ocr_for_single_line(cropped_img)
        print(result)










if __name__ == '__main__':
    main()