% Extract audio features of a single waveform.
% author: SSH'22
close all; clc; clear all;
path = ['../../../data/spcup_2022_training_part1/spcup_2022_training_part1/']
files = dir([path '*.wav']);
filename = [path files(1).name];
[audioIn,fs] = audioread(filename);


aFE = audioFeatureExtractor("SampleRate",fs, ...
    "SpectralDescriptorInput","barkSpectrum", ...
    "spectralCentroid",true, ...
    "spectralKurtosis",true, ...
    "pitch",true, ...
    "linearSpectrum", true, ...
    "melSpectrum", true, ...
    "mfcc", true)

% linearSpectrum, melSpectrum, barkSpectrum, erbSpectrum, mfcc, mfccDelta
%      mfccDeltaDelta, gtcc, gtccDelta, gtccDeltaDelta, spectralCrest, spectralDecrease
%      spectralEntropy, spectralFlatness, spectralFlux, spectralRolloffPoint, spectralSkewness, spectralSlope
%      spectralSpread, harmonicRatio


features = extract(aFE,audioIn);
features = (features - mean(features,1))./std(features,[],1);

idx = info(aFE);
duration = size(audioIn,1)/fs;

subplot(3,1,1)
t = linspace(0,duration,size(audioIn,1));
plot(t,audioIn)

subplot(3,1,2)
t = linspace(0,duration,size(features,1));
plot(t,features(:,idx.spectralCentroid), ...
     t,features(:,idx.spectralKurtosis), ...
     t,features(:,idx.pitch) );
legend("Spectral Centroid","Spectral Kurtosis", "Pitch")
xlabel("Time (s)")

subplot(3,1,3)
t = linspace(0,duration,size(features,1));
plot(t,features(:,idx.melSpectrum));
legend("MelSpectrum")
xlabel("Time (s)")

figure;
melF = features(:, idx.melSpectrum);
surf(melF);

save('features/features1', 'features');

% TODO: 
% - Generate features for all wav files
% - Read mat files in Python
% - Find a way how to seperate different features in Python
% - Find out which features are best: see the link below
% https://se.mathworks.com/help/audio/ug/sequential-feature-selection-for-audio-features.html
