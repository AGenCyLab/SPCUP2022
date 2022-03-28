function [] = compress(ffmpeg_path, audio_path, output_path, bitrate)
    if nargin < 4
        bitrate = 6;
    end
    
    cmd = sprintf('%s -y -i %s -b:a %dk %s', ffmpeg_path, audio_path, bitrate, output_path);
    system(cmd);
end

