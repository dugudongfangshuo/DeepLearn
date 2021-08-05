import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

hparams_file, overrides = 'train.yaml',''
PATH = './results/4234/save/CKPT+2021-04-17+16-05-06+00/model.ckpt'
# 加载超参数文件
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# 加载模型
model=hparams["model"]
model=model.eval()
state_dict = torch.load(PATH)
model.load_state_dict(state_dict)


# 输入一个音频文件
wav = ".\\data\\LibriSpeech\\test-clean\\1089\\134686\\1089-134686-0000.flac"
# 生成噪音文件
def generat_noisy(wav):
    clean_sig = sb.dataio.dataio.read_audio(wav)
    noisy_sig = hparams["env_corruption"](
        clean_sig.unsqueeze(0), torch.ones(1)
    ).squeeze(0)
    return noisy_sig
noisy_wav = generat_noisy(wav)
# 保存噪音文件
tmpfile = './noisy.wav'
sb.dataio.dataio.write_audio(tmpfile, noisy_wav, 16000)
# 计算特征值
def compute_feats(wavs):
    """Returns corresponding log-spectral features of the input waveforms.

    Arguments
    ---------
    wavs : torch.Tensor
        The batch of waveforms to convert to log-spectral features.
    """

    # Log-spectral features
    feats = hparams['compute_STFT'](wavs)
    feats = sb.processing.features.spectral_magnitude(feats, power=0.5)

    # Log1p reduces the emphasis on small differences
    feats = torch.log1p(feats)

    return feats
noisy_wav = noisy_wav.unsqueeze(0)
inputdata = compute_feats(noisy_wav)

# 输入模型
with torch.no_grad():
    output = model(inputdata)

# 转为音频
predict_spec = torch.mul(output, inputdata)

# 还原原始的音频信号
predict_wav =hparams['resynth'](
    torch.expm1(predict_spec), noisy_wav
)
predict_wav = predict_wav.squeeze(0)
# 保存增强后的文件
tmpfile_au = './agument.wav'
sb.dataio.dataio.write_audio(tmpfile_au, predict_wav, 16000)



