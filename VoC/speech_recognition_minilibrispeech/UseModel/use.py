from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_model")
audio_file = 'speechbrain/asr-crdnn-rnnlm-librispeech/example.wav'
a = asr_model.transcribe_file(audio_file)