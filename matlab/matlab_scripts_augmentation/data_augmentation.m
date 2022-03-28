clc
close all
clear

%% IEEE Signal Processing Cup 2022 (SP Cup) - Data augmentation code
% 
% Code to perform data augmentation on SPcup 2022 data
% 
% 3 different types of augmentation are possible:
%      1) Noise injection 
%      2) Add reverberation
%      3) MP3 compression
% 
% Parameterization can be used to obtain different augmentation
% configurations
% 
% Image and Sound Processing Lab - Politecnico di Milano
% Multimedia and Information Security Lab - Drexel University
% 
% Copyright 2022, All rights reserved
%

%% Parameters definition

audio_path = 'test_audio.wav';
ffmpeg_path = '';  % PUT YOUR FFMPEG PATH HERE!
% ffmpeg_path = '/usr/local/bin/';  % Example for OSX

[audioIn, fs] = audioread(audio_path);


%% Noise injection

noise_probability = 1;
SNR_value = 20;

augmenter = audioDataAugmenter( ...
	"AugmentationParameterSource","specify", ...
    "AddNoiseProbability", noise_probability, ...
    "SNR", SNR_value, ...
    "ApplyTimeStretch", false,...
    "ApplyVolumeControl", false, ...
    "ApplyPitchShift", false, ...
    "ApplyTimeStretch", false, ...
    "ApplyTimeShift", false);


data = augment(augmenter, audioIn, fs);
audioAug = data.Audio{1};

[pathstr, name, ext] = fileparts(audio_path);
if isempty(pathstr)
    pathstr = '.';
end
output_path = sprintf('%s/%s_noise%s', pathstr, name, ext);

audiowrite(output_path, audioAug, fs);


%% Add reverberation

predelay = 0;
high_cf = 20000;
diffusion = 0.5;
decay = 0.5;
hifreq_damp = 0.9;
wetdry_mix = 0.25;
fsamp = 16000;

reverb = reverberator( ...
	"PreDelay", predelay, ...
	"HighCutFrequency", high_cf, ...
	"Diffusion", diffusion, ...
	"DecayFactor", decay, ...
    "HighFrequencyDamping", hifreq_damp, ...
	"WetDryMix", wetdry_mix, ...
	"SampleRate", fsamp);

audioRev = reverb(audioIn);
% Stereo to mono
audioRev = .5*(audioRev(:,1) + audioRev(:,2));

[pathstr, name, ext] = fileparts(audio_path);
if isempty(pathstr)
    pathstr = '.';
end
output_path = sprintf('%s/%s_reverb%s', pathstr, name, ext);

audiowrite(output_path, audioRev, fs);


%% Add compression (using ffmpeg)

bitrate = 6;

[pathstr, name, ~] = fileparts(audio_path);
if isempty(pathstr)
    pathstr = '.';
end
output_path = sprintf('%s/%s_compressed.mp3', pathstr, name);

cmd = sprintf('%sffmpeg -y -i %s -b:a %dk %s', ffmpeg_path, audio_path, bitrate, output_path);
system(cmd);

