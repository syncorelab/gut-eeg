function EEG_bp = make_bipolar(EEG_in, pairs)
% make_bipolar – Create bipolar derivations from electrode pairs
%
% UPDATED: Works with mapped EGG# channel labels.
%
% Usage:
%   EEG_bp = make_bipolar(EEG_in, pairs)
%
% EEG_in = EEGLAB dataset (epoched or continuous, with mapped EGG# labels)
% pairs  = cell array of {ch1, ch2} label pairs
%          e.g. {{'EGG1','EGG5'}, {'EGG3','EGG7'}}
%
% Returns a new dataset where each channel is ch1 - ch2.
% Gracefully skips missing channel pairs with a warning.

ndims_data = ndims(EEG_in.data);

if ndims_data == 3
    bp_data   = zeros(length(pairs), EEG_in.pnts, EEG_in.trials);
else
    bp_data   = zeros(length(pairs), EEG_in.pnts);
end
bp_labels = cell(1, length(pairs));

pair_count = 0;

for p = 1:length(pairs)
    ch1 = find(strcmp({EEG_in.chanlocs.labels}, pairs{p}{1}));
    ch2 = find(strcmp({EEG_in.chanlocs.labels}, pairs{p}{2}));

    if isempty(ch1) || isempty(ch2)
        warning('make_bipolar: pair {%s, %s} — channel not found, skipping.', ...
                pairs{p}{1}, pairs{p}{2});
        continue;
    end

    pair_count = pair_count + 1;

    if ndims_data == 3
        bp_data(pair_count,:,:) = EEG_in.data(ch1,:,:) - EEG_in.data(ch2,:,:);
    else
        bp_data(pair_count,:) = EEG_in.data(ch1,:) - EEG_in.data(ch2,:);
    end
    bp_labels{pair_count} = [pairs{p}{1} '-' pairs{p}{2}];
end

% Trim to actual number of valid pairs
valid_pairs = pair_count;
if valid_pairs == 0
    error('make_bipolar:NoPairsFound', ...
        'No valid channel pairs could be created.');
end

if valid_pairs < length(pairs)
    if ndims_data == 3
        bp_data = bp_data(1:valid_pairs, :, :);
    else
        bp_data = bp_data(1:valid_pairs, :);
    end
    bp_labels = bp_labels(1:valid_pairs);
end

EEG_bp = EEG_in;
EEG_bp.data   = bp_data;
EEG_bp.nbchan = length(bp_labels);

% Clear ICA fields — channel space has changed, old weights are invalid
EEG_bp.icaweights = [];
EEG_bp.icasphere  = [];
EEG_bp.icawinv    = [];
EEG_bp.icaact     = [];

% Rebuild chanlocs
for ch = 1:length(bp_labels)
    EEG_bp.chanlocs(ch).labels = bp_labels{ch};
    EEG_bp.chanlocs(ch).X = 0;
    EEG_bp.chanlocs(ch).Y = 0;
    EEG_bp.chanlocs(ch).Z = 0;
end
EEG_bp.chanlocs = EEG_bp.chanlocs(1:length(bp_labels));

EEG_bp = eeg_checkset(EEG_bp);
end
