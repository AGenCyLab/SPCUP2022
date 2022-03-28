function [audioRev] = add_reverbation(audioIn, predelay, high_cf, diffusion, decay, hifreq_damp, wetdry_mix, fsamp)
    if nargin < 2
        predelay = 0;
        high_cf = 20000;
        diffusion = 0.5;
        decay = 0.5;
        hifreq_damp = 0.9;
        wetdry_mix = 0.25;
        fsamp = 16000;
    end

    reverb = reverberator( ...
        "PreDelay", predelay, ...
        "HighCutFrequency", high_cf, ...
        "Diffusion", diffusion, ...
        "DecayFactor", decay, ...
        "HighFrequencyDamping", hifreq_damp, ...
        "WetDryMix", wetdry_mix, ...
        "SampleRate", fsamp);
    
    audioRev = reverb(audioIn);
    audioRev = .5*(audioRev(:,1) + audioRev(:,2));
end

