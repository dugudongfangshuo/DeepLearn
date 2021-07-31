import torchaudio
from speechbrain.pretrained import EncoderClassifier


classifier = EncoderClassifier.from_hparams(source="/content/best_model/", hparams_file='hparams_inference.yaml', savedir="/content/best_model/")

# Perform classification
audio_file = 'data/LibriSpeech/train-clean-5/5789/70653/5789-70653-0036.flac'
signal, fs = torchaudio.load(audio_file) # test_speaker: 5789
output_probs, score, index, text_lab = classifier.classify_batch(signal)
print('Target: 5789, Predicted: ' + text_lab[0])

# Another speaker
audio_file = 'data/LibriSpeech/train-clean-5/460/172359/460-172359-0012.flac'
signal, fs =torchaudio.load(audio_file) # test_speaker: 460
output_probs, score, index, text_lab = classifier.classify_batch(signal)
print('Target: 460, Predicted: ' + text_lab[0])

# And if you want to extract embeddings...
embeddings = classifier.encode_batch(signal)
