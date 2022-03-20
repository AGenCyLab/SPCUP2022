clear;
close all;
clc;

%% Parameters

% fixed params
rng(0, 'twister');

ffmpeg_path = 'C:\ffmpeg\bin\ffmpeg.exe';  % PUT YOUR FFMPEG PATH HERE!
noise_probability = 1;
predelay = 1;
high_cf = 20000;
fsamp = 16000;

%% These three should be changed accordingly before running the script
path_to_dataset_root = "C:\Users\Phantomhive\Documents\Code\SPCUP22\SPCUP2022\data\raw_audio\spcup22\training\part2\spcup_2022_unseen";
annotations_filename = "labels.csv";
output_dataset_root = "C:\Users\Phantomhive\Documents\Code\SPCUP22\SPCUP2022\data\raw_audio\spcup22\training\part2_aug\spcup_2022_unseen";
mkdir(output_dataset_root);

%% read csv
annotations_filepath = fullfile(path_to_dataset_root, annotations_filename);
csv = readtable(annotations_filepath);

% 3 transformations on every file = 3 times more files
modified_csv_matrix = cell((size(csv, 1) * 3) + 1, 2);
modified_csv_matrix{1, 1} = csv.Properties.VariableNames{1};
modified_csv_matrix{1, 2} = csv.Properties.VariableNames{2};

% start from 2 since we just added header row
modified_csv_matrix_index = 2; 

%% Loop through the csv
for index = 1:size(csv, 1)
    % randomized parameters, kept within the threshold of values
    % in the original sample script
    bitrate = randsample(4:8, 1);
    SNR_value = randsample(10:30, 1); 
    diffusion = (0.6 - 0.4).*rand(1, 1) + 0.4;
    decay = (0.6 - 0.4).*rand(1, 1) + 0.4;
    hifreq_damp = (0.95 - 0.6).*rand(1, 1) + 0.6;
    wetdry_mix = (0.5 - 0.25).*rand(1, 1) + 0.25;
    
    filename = csv.track(index);
    label = csv.algorithm(index);
    
    audio_path = fullfile(path_to_dataset_root, filename);
    
    %% read file
    [audioIn, fs] = audioread(audio_path);
    
    %% Add noise and add to modified_csv
    audioAug = noise_inject(audioIn, fs, noise_probability, SNR_value);
    [output_path, filename_with_ext] = construct_output_filename(audio_path, output_dataset_root, "noise");
    audiowrite(output_path, audioAug, fs);
    modified_csv_matrix{modified_csv_matrix_index, 1} = filename_with_ext;
    modified_csv_matrix{modified_csv_matrix_index, 2} = label;
    modified_csv_matrix_index = modified_csv_matrix_index + 1;
    
    %% add reverbation and add to modified_csv
    audioRev = add_reverbation(audioIn, predelay, high_cf, diffusion, decay, hifreq_damp, wetdry_mix, fsamp);
    [output_path, filename_with_ext] = construct_output_filename(audio_path, output_dataset_root, "reverb");
    audiowrite(output_path, audioRev, fs);
    modified_csv_matrix{modified_csv_matrix_index, 1} = filename_with_ext;
    modified_csv_matrix{modified_csv_matrix_index, 2} = label;
    modified_csv_matrix_index = modified_csv_matrix_index + 1;
    
    %% add compression and add to modified_csv
    [output_path, filename_with_ext] = construct_output_filename(audio_path, output_dataset_root, "compress");
    compress(ffmpeg_path, audio_path, output_path, bitrate);
    modified_csv_matrix{modified_csv_matrix_index, 1} = filename_with_ext;
    modified_csv_matrix{modified_csv_matrix_index, 2} = label;
    modified_csv_matrix_index = modified_csv_matrix_index + 1;
    
end

%% write to csv file
fid = fopen(fullfile(output_dataset_root, 'labels_aug.csv'), 'w');
fprintf(fid,'%s, %s\n', modified_csv_matrix{1,:});
for index=2:size(modified_csv_matrix, 1)
    fprintf(fid,'%s, %d\n', modified_csv_matrix{index, :});
end
fclose(fid);
