function [output_path, filename_with_ext] = construct_output_filename(audio_path, output_dataset_root, augment_type)
    % augment_type: one of ("noise", "reverb", "compress"    
    [pathstr, name, ext] = fileparts(audio_path);
    filename_with_ext = sprintf('%s_%s%s', name, augment_type, ext);
    output_path = fullfile(output_dataset_root, filename_with_ext);
end

