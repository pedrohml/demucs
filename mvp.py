import os
import torch
import numpy as np
from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.htdemucsv2 import HTDemucsV2
from demucs.pretrained import get_model
from demucs.separate import load_track

device = torch.device('mps')
shifts = 1
overlap = 0.25
split = True
workers = os.cpu_count()
segment = None

bag_of_models = get_model('htdemucs')
htmodel = bag_of_models.models[0]

track = load_track('./sample.mp4', bag_of_models.audio_channels, bag_of_models.samplerate)

ref = track.mean(dim=0)
track -= ref.mean()
track /= ref.std()

new_model = HTDemucsV2.from_v1(htmodel)
new_model.eval()

sources = apply_model(new_model, track[None], device=device, shifts=shifts,
                      split=split, overlap=overlap, progress=True,
                      num_workers=workers, segment=segment)[0]

sources *= ref.std()
sources += ref.mean()

save_metadata = {
    'samplerate': new_model.samplerate,
    'bitrate': 320,
    'preset': 2,
    'clip': 'rescale',
    'as_float': False,
    'bits_per_sample': 16,
}

save_audio(sources[-1], 'sample_rec.mp3', **save_metadata)