function qc = epoch_quality(EEG_in)
% epoch_quality – Compute per-epoch quality metrics (flag but do not reject)
%
% Usage:
%   qc = epoch_quality(EEG_in)
%
% Returns a struct with fields:
%   max_amp       – max absolute amplitude per epoch
%   rms           – RMS amplitude per epoch
%   kurtosis_val  – mean kurtosis across channels per epoch
%   spec_flat     – spectral flatness per epoch (high = noise-like)
%   flag          – logical vector flagging epochs above 95th percentile

qc = struct();

qc.max_amp      = squeeze(max(max(abs(EEG_in.data), [], 1), [], 2));
qc.rms          = squeeze(sqrt(mean(mean(EEG_in.data.^2, 1), 2)));
qc.kurtosis_val = squeeze(mean(kurtosis(EEG_in.data, [], 2), 1));

% Spectral flatness per epoch (high = noise-like)
qc.spec_flat = zeros(EEG_in.trials, 1);
for ep = 1:EEG_in.trials
    psd = mean(abs(fft(EEG_in.data(:,:,ep), [], 2)).^2, 1);
    psd = psd(1:floor(end/2));
    qc.spec_flat(ep) = exp(mean(log(psd + eps))) / mean(psd);
end

% Flag but do NOT reject
qc.flag = qc.max_amp > prctile(qc.max_amp, 95) | ...
          qc.kurtosis_val > prctile(qc.kurtosis_val, 95);

fprintf('  Flagged: %d / %d epochs (%.1f%%)\n', ...
    sum(qc.flag), EEG_in.trials, 100 * mean(qc.flag));
end
