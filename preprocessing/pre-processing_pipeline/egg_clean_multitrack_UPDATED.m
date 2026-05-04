function [] = egg_clean_multitrack(inpath, outpath, filename)
% egg_clean_multitrack – Multi-track cleaning pipeline for abdominal EGG
%
% UPDATED: Channel mapping, C4 reference track, and consistent remapping.
%
% Usage:
%   egg_clean_multitrack(inpath, outpath, filename)
%
% inpath   = directory containing the .vhdr file
% outpath  = directory for all output files (created if missing)
% filename = name of the .vhdr file (e.g. '019.vhdr')
%
% Pipeline overview:
%
%   Step 0 — Import, channel mapping, and preparation
%            Load BrainVision, map raw channel names to canonical EGG1-EGG19,
%            assign 2D layout, verify events.
%
%   Step 1 — Track A: minimal clean (broadband, DC–1024 Hz)
%            CleanLine 50 Hz removal, bad channel detection/interpolation,
%            gross artifact flagging, epoching [-2 18]s, export.
%
%   Step 2 — ICA decomposition (for Tracks B & C)
%            1 Hz highpass copy, downsample to 256 Hz, extended Infomax ICA,
%            transfer weights to original data, semi-automatic component
%            identification (cardiac, respiratory, movement).
%
%   Step 3 — Tracks B1–B4: frequency-band extraction after ICA cleaning
%            B1 = gastric slow wave (0.0083–0.15 Hz, 32 Hz)
%            B2 = intestinal slow wave (0.0083–0.63 Hz, 32 Hz)
%            B3 = neural burst band (1–200 Hz, 512 Hz)
%            B4 = autonomic modulation (0.0083–30 Hz, 512 Hz)
%
%   Step 4 — Tracks C: re-referencing variants
%            C1 = average reference
%            C2 = bipolar vertical derivations
%            C3 = bipolar horizontal derivations
%            C4 = single-channel reference to EGG16 (raw C1)
%
%   Step 5 — Epoch quality control (flag, do not reject)
%
%   Step 5.5 — Optional across-subject normalization (per track, per channel)
%
%   Step 6 — Export all tracks to .mat for deep learning
%
% Requires: EEGLAB with bva-io and cleanline plugins on the MATLAB path.

%% ========================================================================
%  0. CHECKS & CONFIG
%  ========================================================================

[~, nameOnly, ext] = fileparts(filename);
if ~strcmpi(ext, '.vhdr')
    error('egg_clean_multitrack:InputFile', ...
        'This function accepts only .vhdr files, got "%s".', filename);
end

if ~exist(outpath, 'dir')
    mkdir(outpath);
end

% Subject-specific output subfolder
subj_outpath = fullfile(outpath, nameOnly);
if ~exist(subj_outpath, 'dir')
    mkdir(subj_outpath);
end

fprintf('\n=== egg_clean_multitrack: %s ===\n', filename);
fprintf('Output: %s\n', subj_outpath);

% Across-subject normalization config
ZSCORE_MODE = 'off';
ZSCORE_FILE = fullfile(outpath, 'across_subject_channel_stats.mat');

%% ========================================================================
%  STEP 0: IMPORT & CHANNEL MAPPING
%  ========================================================================
fprintf('\n--- Step 0: Import & channel mapping ---\n');

[ALLEEG, EEG, CURRENTSET] = eeglab('nogui'); %#ok<ASGLU>

% 0.1 Load BrainVision file
EEG = pop_loadbv(inpath, filename);
EEG = eeg_checkset(EEG);

if EEG.srate ~= 2048
    error('egg_clean_multitrack:SampleRate', ...
        'Expected 2048 Hz, got %d Hz. Adjust pipeline for this rate or exclude file.', EEG.srate);
end

fprintf('Loaded raw data: %d channels, %d points, %.1f seconds\n', ...
    EEG.nbchan, EEG.pnts, EEG.xmax);

% 0.2 Channel mapping: map raw labels to canonical EGG1-EGG19
fprintf('\nPerforming channel mapping...\n');

% Define canonical raw channel name list (in canonical order)
canonical_raw_names = {
    'Fp1',  'AF4',   'AF7',   'F1',   'F4', ...       % EGG1-5
    'F2',   'AF8',  'F7',  'F3', ...              % EGG6-9
    'C1',   'FCz',   'F8',   'F5', ...               % EGG10-13
    'FC3',  'FT7',  'C2',   'C4', ...               % EGG14-17
    'F6',  'FT8'};                                 % EGG18-19

egglab_names = arrayfun(@(i) sprintf('EGG%d', i), 1:19, 'UniformOutput', false);

% Get raw channel labels from loaded file
raw_labels = {EEG.chanlocs.labels};

% Find indices of canonical channels in raw data
[mapping, omitted_chans, c1_idx_raw] = map_channels_to_canonical(raw_labels, canonical_raw_names);

% Check if all required channels were found
if any(isnan(mapping))
    missing_idx = find(isnan(mapping));
    missing_names = canonical_raw_names(missing_idx);
    error('egg_clean_multitrack:MissingChannels', ...
        'Missing required channels:\n  %s', ...
        sprintf('%s, ', missing_names{:}));
end

% Check for C1 (used as reference for C4 track)
if isnan(c1_idx_raw)
    error('egg_clean_multitrack:MissingC1', ...
        'Required reference channel C1 (maps to EGG16) not found in raw data.');
end

% Reorder and rename channels
EEG = pop_select(EEG, 'channel', mapping);
EEG = eeg_checkset(EEG);

% Verify exactly 19 channels after selection
if EEG.nbchan ~= 19
    error('egg_clean_multitrack:ChannelCountMismatch', ...
        'After mapping, expected 19 channels but got %d.', EEG.nbchan);
end

% Assign canonical labels
fprintf('Channel mapping summary:\n');
fprintf('  Raw label -> EGG#\n');
for ch = 1:19
    EEG.chanlocs(ch).labels = egglab_names{ch};
    fprintf('    %s -> %s\n', canonical_raw_names{ch}, egglab_names{ch});
end

if ~isempty(omitted_chans)
    fprintf('\nOmitted channels (not in canonical 19):\n');
    for i = 1:length(omitted_chans)
        fprintf('  %s\n', omitted_chans{i});
    end
end

fprintf('\nReference channel for C4 track: C1 -> EGG10 (index %d after mapping)\n', 16);

% 0.3 Assign 2D layout to the 19 mapped channels
fprintf('\nAssigning 2D layout...\n');

xy = [
  -0.15  0.8;   0.15  0.8;                                               % EGG1-2
  -0.25  0.5;  -0.10  0.5;   0.0  0.5;  0.10  0.5;  0.25  0.5;          % EGG3-7
  -0.45  0.2;  -0.15  0.2;   0.0  0.2;  0.15  0.2;  0.45  0.2;          % EGG8-12
  -0.50 -0.1;  -0.30 -0.1;  -0.10 -0.1;  0.10 -0.1;  0.30 -0.1;  0.50 -0.1;  % EGG13-18
   0.0  -0.4;                                                             % EGG19
];

for ch = 1:19
    EEG.chanlocs(ch).X = xy(ch, 1);
    EEG.chanlocs(ch).Y = xy(ch, 2);
    EEG.chanlocs(ch).Z = 0;
end
EEG = eeg_checkset(EEG);

fprintf('Assigned 2D layout to 19 mapped channels.\n');

% 0.4 Verify event structure
fprintf('\nVerifying event structure...\n');
events = {EEG.event.type};
unique_events = unique(events);
fprintf('Event types found: ');
disp(unique_events);

for trig = [1 2 4]
    trig_str = num2str(trig);
    n = sum(strcmp(events, trig_str) | strcmp(events, ['S ' trig_str]) | ...
            strcmp(events, ['S  ' trig_str]) | strcmp(events, ['s' trig_str]) | ...
            strcmp(events, ['s ' trig_str]) | strcmp(events, ['s  ' trig_str]));
    fprintf('  Trigger %d: %d trials\n', trig, n);
end

% Store original channel locations and mapping info
EEG.etc.original_chanlocs = EEG.chanlocs;
EEG.etc.channel_mapping = struct(...
    'raw_names', {canonical_raw_names}, ...
    'mapped_names', {egglab_names}, ...
    'omitted_channels', {omitted_chans}, ...
    'c1_reference_index', 10);

% Save the shared base dataset
EEG_base = EEG;

%% ========================================================================
%  STEP 1: TRACK A — MINIMAL CLEAN (BROADBAND)
%  ========================================================================
fprintf('\n--- Step 1: Track A — Minimal clean (broadband) ---\n');

EEG_A = EEG_base;

% 1.1 CleanLine: remove 50 Hz + harmonics
fprintf('Running CleanLine (50 Hz + harmonics)...\n');
try
    EEG_A = pop_cleanline(EEG_A, 'linefreqs', [50 100 150 200 250], ...
        'scanforlines', 1, 'bandwidth', 2, 'sigtype', 'Channels', ...
        'computepower', 1, 'normtype', 'zscore');
catch ME
    warning('egg_clean_multitrack:CleanLine', ...
            'CleanLine failed (%s). Using notch filter as fallback.', ME.message);
    for fnotch = [50 100 150 200 250]
        if fnotch < EEG_A.srate / 2
            EEG_A = pop_eegfiltnew(EEG_A, 'locutoff', fnotch - 1, ...
                'hicutoff', fnotch + 1, 'revfilt', 1);
        end
    end
end

% 1.2 Bad channel detection
fprintf('Detecting bad channels...\n');
EEG_temp = pop_reref(EEG_A, []);

chan_var  = var(EEG_temp.data, [], 2);
chan_kurt = kurtosis(EEG_temp.data, [], 2);
chan_corr = zeros(EEG_temp.nbchan, 1);

for ch = 1:EEG_temp.nbchan
    others = setdiff(1:EEG_temp.nbchan, ch);
    r = corrcoef(EEG_temp.data(ch,:)', mean(EEG_temp.data(others,:), 1)');
    chan_corr(ch) = r(1,2);
end

var_z  = (chan_var  - mean(chan_var))  / std(chan_var);
kurt_z = (chan_kurt - mean(chan_kurt)) / std(chan_kurt);
bad_mask = abs(var_z) > 3 | abs(kurt_z) > 3 | chan_corr < 0.3;
bad_chans = {EEG_A.chanlocs(bad_mask).labels};

if ~isempty(bad_chans)
    fprintf('  Flagged channels: ');
    disp(bad_chans);
else
    fprintf('  No bad channels flagged.\n');
end

% 1.3 Interpolate bad channels (max 3)
if sum(bad_mask) > 3
    warning('egg_clean_multitrack:TooManyBadChannels', ...
            'More than 3 bad channels detected. Limiting interpolation to 3.');
    [~, idx] = sort(abs(var_z) + abs(kurt_z), 'descend');
    bad_mask(:) = 0;
    bad_mask(idx(1:3)) = 1;
    bad_chans = {EEG_A.chanlocs(bad_mask).labels};
end

if any(bad_mask)
    EEG_A = pop_interp(EEG_A, find(bad_mask), 'spherical');
    fprintf('  Interpolated: ');
    disp(bad_chans);
end

% Apply cleaned version to ICA branch too
EEG_cleaned = EEG_A;

% 1.4 Gross artifact rejection (flag only)
threshold_uV = 500;
bad_samples = any(abs(EEG_A.data) > threshold_uV, 1);
expand_pts = round(0.5 * EEG_A.srate);
bad_expanded = conv(double(bad_samples), ones(1, 2*expand_pts+1), 'same') > 0;
fprintf('  Gross artifact: %.2f%% of data\n', 100 * mean(bad_expanded));
EEG_A.etc.track_A_rejected_pct = 100 * mean(bad_expanded);

% 1.5 Epoch Track A
trig_types = get_trigger_strings(EEG_A);
fprintf('  Epoching with triggers: ');
disp(trig_types);

EEG_A = pop_epoch(EEG_A, trig_types, [-1 15]);
EEG_A = eeg_checkset(EEG_A);
fprintf('  Track A epochs: %d\n', EEG_A.trials);

EEG_A = pop_rmbase(EEG_A, [-1000 0]);

EEG_A.setname = [nameOnly '_Track_A_broadband'];
pop_saveset(EEG_A, 'filename', [nameOnly '_Track_A_broadband.set'], ...
    'filepath', subj_outpath);
fprintf('  Saved Track A .set\n');

%% ========================================================================
%  STEP 2: ICA DECOMPOSITION (for Tracks B & C)
%  ========================================================================
fprintf('\n--- Step 2: ICA decomposition ---\n');

EEG_ica = pop_eegfiltnew(EEG_cleaned, 'locutoff', 1, 'plotfreqz', 0);
EEG_ica = pop_resample(EEG_ica, 256);

icaRank = rank(double(EEG_ica.data(:,:)));
fprintf('  Data rank: %d, channels: %d\n', icaRank, EEG_ica.nbchan);

if icaRank < EEG_ica.nbchan
    warning('egg_clean_multitrack:RankDeficient', ...
            'Data rank (%d) < channels (%d). Running reduced ICA.', ...
            icaRank, EEG_ica.nbchan);
    EEG_ica = pop_runica(EEG_ica, 'icatype', 'runica', 'extended', 1, 'pca', icaRank);
else
    EEG_ica = pop_runica(EEG_ica, 'icatype', 'runica', 'extended', 1);
end

EEG_forB = EEG_cleaned;
EEG_forB.icaweights = EEG_ica.icaweights;
EEG_forB.icasphere  = EEG_ica.icasphere;
if isfield(EEG_ica, 'icawinv')
    EEG_forB.icawinv = EEG_ica.icawinv;
end
EEG_forB = eeg_checkset(EEG_forB);

ncomp = size(EEG_forB.icaweights, 1);
fprintf('  ICA components: %d\n', ncomp);

% Semi-automatic artifact detection (heuristic, NO ICLABEL)
artifact_comps  = [];
artifact_labels = {};

icaact = eeg_getica(EEG_forB);

for ic = 1:ncomp
    ic_data = icaact(ic, :);

    nfft = min(2^nextpow2(length(ic_data)), 2^16);
    [psd, f] = pwelch(double(ic_data), nfft/2, nfft/4, nfft, EEG_forB.srate);

    % Cardiac detection
    cardiac_band = f >= 0.8 & f <= 2.0;
    broad_band   = f >= 0.1 & f <= 30;
    cardiac_power = max(psd(cardiac_band));
    broad_mean    = mean(psd(broad_band));
    cardiac_ratio = cardiac_power / (broad_mean + eps);

    max_lag = min(round(2 * EEG_forB.srate), length(ic_data) - 1);
    ac = xcorr(double(ic_data), max_lag, 'coeff');
    ac = ac(max_lag+1:end);
    lag_range = round(0.5 * EEG_forB.srate):round(1.5 * EEG_forB.srate);
    lag_range = lag_range(lag_range <= length(ac));
    if ~isempty(lag_range)
        ac_peak = max(ac(lag_range));
    else
        ac_peak = 0;
    end

    if cardiac_ratio > 5 && ac_peak > 0.3
        artifact_comps(end+1)  = ic; %#ok<AGROW>
        artifact_labels{end+1} = 'cardiac'; %#ok<AGROW>
        fprintf('    IC%d -> CARDIAC (ratio=%.1f, ac_peak=%.2f)\n', ic, cardiac_ratio, ac_peak);
        continue;
    end

    % Respiratory detection
    resp_band = f >= 0.15 & f <= 0.5;
    resp_power = mean(psd(resp_band));
    other_power = mean(psd(broad_band & ~resp_band));
    resp_ratio = resp_power / (other_power + eps);

    topo = EEG_forB.icawinv(:, ic);
    ch_labels = {EEG_forB.chanlocs.labels};
    
    % Diaphragm-associated channels (mapped EGG# labels)
    diaphragm_chs = find(ismember(ch_labels, {'EGG1','EGG7','EGG11','EGG2'}));
    if ~isempty(diaphragm_chs)
        diaphragm_loading = mean(abs(topo(diaphragm_chs)));
        other_loading = mean(abs(topo(setdiff(1:length(topo), diaphragm_chs))));
        topo_ratio = diaphragm_loading / (other_loading + eps);
    else
        topo_ratio = 0;
    end

    if resp_ratio > 3 && topo_ratio > 1.5
        artifact_comps(end+1)  = ic; %#ok<AGROW>
        artifact_labels{end+1} = 'respiratory'; %#ok<AGROW>
        fprintf('    IC%d -> RESPIRATORY (ratio=%.1f, topo=%.1f)\n', ic, resp_ratio, topo_ratio);
        continue;
    end

    % Movement detection (high-frequency + lateral loading)
    hf_band = f >= 40 & f <= min(200, EEG_forB.srate/2 - 1);
    lf_band = f >= 1 & f <= 30;
    if any(hf_band) && any(lf_band)
        hf_power = mean(psd(hf_band));
        lf_power = mean(psd(lf_band));
        hf_ratio = hf_power / (lf_power + eps);

        % Lateral-associated channels (mapped EGG# labels)
        lateral_chs = find(ismember(ch_labels, {'EGG3','EGG14','EGG18','EGG13'}));
        if ~isempty(lateral_chs)
            lateral_loading = mean(abs(topo(lateral_chs)));
            central_chs = setdiff(1:length(topo), lateral_chs);
            central_loading = mean(abs(topo(central_chs)));
            lat_ratio = lateral_loading / (central_loading + eps);
        else
            lat_ratio = 0;
        end

        if hf_ratio > 2 && lat_ratio > 2
            artifact_comps(end+1)  = ic; %#ok<AGROW>
            artifact_labels{end+1} = 'movement'; %#ok<AGROW>
            fprintf('    IC%d -> MOVEMENT (hf_ratio=%.1f, lat=%.1f)\n', ic, hf_ratio, lat_ratio);
        end
    end
end

if length(artifact_comps) > 4
    warning('Detected %d artifact components — limiting to 4 most confident.', ...
            length(artifact_comps));
    artifact_comps  = artifact_comps(1:4);
    artifact_labels = artifact_labels(1:4);
end

fprintf('  Artifact components to remove: %d\n', length(artifact_comps));
for i = 1:length(artifact_comps)
    fprintf('    IC%d = %s\n', artifact_comps(i), artifact_labels{i});
end

EEG_forB.etc.ica_removed = struct('components', artifact_comps, ...
    'labels', {artifact_labels}, 'method', 'semi-automatic heuristic (no ICLabel)');

if ~isempty(artifact_comps)
    EEG_forB = pop_subcomp(EEG_forB, artifact_comps, 0);
end
EEG_forB = eeg_checkset(EEG_forB);

fprintf('  Components remaining: %d\n', ...
    size(EEG_forB.icaweights, 1) - length(artifact_comps));

%% ========================================================================
%  STEP 3: TRACKS B1–B4 — FREQUENCY-BAND EXTRACTION
%  ========================================================================
fprintf('\n--- Step 3: Tracks B1–B4 — Frequency bands ---\n');

trig_types = get_trigger_strings(EEG_forB);

EEG_slow = pop_resample(EEG_forB, 32);
EEG_fast = pop_resample(EEG_forB, 512);

fprintf('  B1: Gastric slow wave (0.0083–0.15 Hz)...\n');
EEG_B1 = pop_eegfiltnew(EEG_slow, 'locutoff', 0.0083, 'hicutoff', 0.15);
EEG_B1 = pop_epoch(EEG_B1, trig_types, [-1 15]);
EEG_B1 = pop_rmbase(EEG_B1, [-1000 0]);
EEG_B1.setname = [nameOnly '_Track_B1_gastric'];
pop_saveset(EEG_B1, 'filename', [nameOnly '_Track_B1_gastric.set'], ...
    'filepath', subj_outpath);
fprintf('    %d epochs, srate=%d Hz\n', EEG_B1.trials, EEG_B1.srate);

fprintf('  B2: Intestinal slow wave (0.0083–0.63 Hz)...\n');
EEG_B2 = pop_eegfiltnew(EEG_slow, 'locutoff', 0.0083, 'hicutoff', 0.63);
EEG_B2 = pop_epoch(EEG_B2, trig_types, [-1 15]);
EEG_B2 = pop_rmbase(EEG_B2, [-1000 0]);
EEG_B2.setname = [nameOnly '_Track_B2_intestinal'];
pop_saveset(EEG_B2, 'filename', [nameOnly '_Track_B2_intestinal.set'], ...
    'filepath', subj_outpath);
fprintf('    %d epochs, srate=%d Hz\n', EEG_B2.trials, EEG_B2.srate);

fprintf('  B3: Neural burst band (20–30 Hz)...\n');
EEG_B3 = pop_eegfiltnew(EEG_fast, 'locutoff', 20, 'hicutoff', 30);
EEG_B3 = pop_epoch(EEG_B3, trig_types, [-1 15]);
EEG_B3 = pop_rmbase(EEG_B3, [-1000 0]);
EEG_B3.setname = [nameOnly '_Track_B3_neural'];
pop_saveset(EEG_B3, 'filename', [nameOnly '_Track_B3_neural.set'], ...
    'filepath', subj_outpath);
fprintf('    %d epochs, srate=%d Hz\n', EEG_B3.trials, EEG_B3.srate);

fprintf('  B4: Autonomic modulation (0.0083–30 Hz)...\n');
EEG_B4 = pop_eegfiltnew(EEG_fast, 'locutoff', 0.0083, 'hicutoff', 30);
EEG_B4 = pop_epoch(EEG_B4, trig_types, [-1 15]);
EEG_B4 = pop_rmbase(EEG_B4, [-1000 0]);
EEG_B4.setname = [nameOnly '_Track_B4_autonomic'];
pop_saveset(EEG_B4, 'filename', [nameOnly '_Track_B4_autonomic.set'], ...
    'filepath', subj_outpath);
fprintf('    %d epochs, srate=%d Hz\n', EEG_B4.trials, EEG_B4.srate);

%% ========================================================================
%  STEP 4: TRACKS C — RE-REFERENCING VARIANTS
%  ========================================================================
fprintf('\n--- Step 4: Tracks C — Re-referencing variants ---\n');

track_names = {'B1', 'B2', 'B3', 'B4'};
track_sets  = {EEG_B1, EEG_B2, EEG_B3, EEG_B4};

% C1: Average reference
fprintf('  C1: Average reference...\n');
for t = 1:length(track_sets)
    EEG_avg = pop_reref(track_sets{t}, []);
    EEG_avg.setname = sprintf('%s_Track_%s_avgref', nameOnly, track_names{t});
    pop_saveset(EEG_avg, ...
        'filename', sprintf('%s_Track_%s_avgref.set', nameOnly, track_names{t}), ...
        'filepath', subj_outpath);
end
fprintf('    Saved 4 avgref .set files\n');

% C2 & C3: Bipolar derivations
vert_pairs = {
    {'EGG1','EGG11'}, {'EGG8','EGG12'}, ...
    {'EGG11','EGG3'}, {'EGG12','EGG15'}, ...
    {'EGG3','EGG14'}, {'EGG16','EGG18'}, ...
    {'EGG18','EGG19'}, ...
};

horiz_pairs = {
    {'EGG1','EGG8'}, ...
    {'EGG7','EGG9'}, ...
    {'EGG11','EGG12'}, ...
    {'EGG2','EGG5'}, ...
    {'EGG10','EGG13'}, ...
    {'EGG14','EGG6'}, ...
};

fprintf('  C2/C3: Bipolar derivations...\n');
for t = 1:length(track_sets)
    EEG_bpv = make_bipolar(track_sets{t}, vert_pairs);
    EEG_bpv.setname = sprintf('%s_Track_%s_bipolar_vert', nameOnly, track_names{t});
    pop_saveset(EEG_bpv, ...
        'filename', sprintf('%s_Track_%s_bipolar_vert.set', nameOnly, track_names{t}), ...
        'filepath', subj_outpath);

    EEG_bph = make_bipolar(track_sets{t}, horiz_pairs);
    EEG_bph.setname = sprintf('%s_Track_%s_bipolar_horiz', nameOnly, track_names{t});
    pop_saveset(EEG_bph, ...
        'filename', sprintf('%s_Track_%s_bipolar_horiz.set', nameOnly, track_names{t}), ...
        'filepath', subj_outpath);
end
fprintf('    Saved 8 bipolar .set files\n');

% C4: Single-channel reference to EGG10 (mapped from raw C1)
fprintf('  C4: Single-channel reference to EGG10 (raw C1)...\n');
for t = 1:length(track_sets)
    EEG_c1ref = pop_reref(track_sets{t}, 10, 'keepref', 'on');
    EEG_c1ref.setname = sprintf('%s_Track_%s_c1ref', nameOnly, track_names{t});
    pop_saveset(EEG_c1ref, ...
        'filename', sprintf('%s_Track_%s_c1ref.set', nameOnly, track_names{t}), ...
        'filepath', subj_outpath);
end
fprintf('    Saved 4 C1-reference .set files\n');

%% ========================================================================
%  STEP 5: EPOCH QUALITY CONTROL
%  ========================================================================
fprintf('\n--- Step 5: Epoch quality control ---\n');

fprintf('  Track A:\n');
EEG_A.etc.epoch_qc = epoch_quality(EEG_A);

fprintf('  Track B1:\n');
EEG_B1.etc.epoch_qc = epoch_quality(EEG_B1);

fprintf('  Track B2:\n');
EEG_B2.etc.epoch_qc = epoch_quality(EEG_B2);

fprintf('  Track B3:\n');
EEG_B3.etc.epoch_qc = epoch_quality(EEG_B3);

fprintf('  Track B4:\n');
EEG_B4.etc.epoch_qc = epoch_quality(EEG_B4);

%% ========================================================================
%  STEP 5.5: ACROSS-SUBJECT NORMALIZATION (per track, per channel)
%  ========================================================================
fprintf('\n--- Step 5.5: Across-subject normalization ---\n');

if strcmpi(ZSCORE_MODE, 'collect')
    update_subject_norm_stats(EEG_A,  'Track_A_broadband', ZSCORE_FILE);
    update_subject_norm_stats(EEG_B1, 'Track_B1_gastric', ZSCORE_FILE);
    update_subject_norm_stats(EEG_B2, 'Track_B2_intestinal', ZSCORE_FILE);
    update_subject_norm_stats(EEG_B3, 'Track_B3_neural', ZSCORE_FILE);
    update_subject_norm_stats(EEG_B4, 'Track_B4_autonomic', ZSCORE_FILE);

    fprintf('Collected across-subject stats for this subject.\n');

elseif strcmpi(ZSCORE_MODE, 'apply')
    EEG_A  = apply_subject_norm(EEG_A,  'Track_A_broadband', ZSCORE_FILE);
    EEG_B1 = apply_subject_norm(EEG_B1, 'Track_B1_gastric', ZSCORE_FILE);
    EEG_B2 = apply_subject_norm(EEG_B2, 'Track_B2_intestinal', ZSCORE_FILE);
    EEG_B3 = apply_subject_norm(EEG_B3, 'Track_B3_neural', ZSCORE_FILE);
    EEG_B4 = apply_subject_norm(EEG_B4, 'Track_B4_autonomic', ZSCORE_FILE);

    track_sets = {EEG_B1, EEG_B2, EEG_B3, EEG_B4};

    fprintf('Applied across-subject normalization to Track A and B tracks.\n');

else
    fprintf('Across-subject normalization disabled.\n');
end

%% ========================================================================
%  STEP 6: EXPORT FOR DEEP LEARNING (.mat)
%  ========================================================================
fprintf('\n--- Step 6: Export for deep learning ---\n');

dl_path = fullfile(subj_outpath, 'dl_export');
if ~exist(dl_path, 'dir')
    mkdir(dl_path);
end

% Track A
export_for_dl(EEG_A, 'Track_A_broadband', dl_path);

% Track B
export_for_dl(EEG_B1, 'Track_B1_gastric', dl_path);
export_for_dl(EEG_B2, 'Track_B2_intestinal', dl_path);
export_for_dl(EEG_B3, 'Track_B3_neural', dl_path);
export_for_dl(EEG_B4, 'Track_B4_autonomic', dl_path);

% Track C1 (average reference)
for t = 1:length(track_sets)
    EEG_avg = pop_reref(track_sets{t}, []);
    EEG_avg.etc.epoch_qc = epoch_quality(EEG_avg);
    export_for_dl(EEG_avg, sprintf('Track_%s_avgref', track_names{t}), dl_path);
end

% Track C2 & C3 (bipolar)
for t = 1:length(track_sets)
    EEG_bpv = make_bipolar(track_sets{t}, vert_pairs);
    export_for_dl(EEG_bpv, sprintf('Track_%s_bipolar_vert', track_names{t}), dl_path);

    EEG_bph = make_bipolar(track_sets{t}, horiz_pairs);
    export_for_dl(EEG_bph, sprintf('Track_%s_bipolar_horiz', track_names{t}), dl_path);
end

% Track C4 (C1 reference)
for t = 1:length(track_sets)
    EEG_c1ref = pop_reref(track_sets{t}, 16, 'keepref', 'on');
    EEG_c1ref.etc.epoch_qc = epoch_quality(EEG_c1ref);
    export_for_dl(EEG_c1ref, sprintf('Track_%s_c1ref', track_names{t}), dl_path);
end

%% ========================================================================
%  SAVE PROCESSING LOG
%  ========================================================================

log = struct();
log.filename          = filename;
log.date              = datestr(now);
log.channel_mapping   = EEG_base.etc.channel_mapping;
log.channels_removed  = EEG_base.etc;
log.ica_decisions     = EEG_forB.etc.ica_removed;
log.bad_channels      = bad_chans;
log.artifact_pct      = EEG_A.etc.track_A_rejected_pct;
log.zscore_mode       = ZSCORE_MODE;
log.zscore_file       = ZSCORE_FILE;
save(fullfile(subj_outpath, [nameOnly '_processing_log.mat']), 'log');

fprintf('\n=== egg_clean_multitrack COMPLETE: %s ===\n', filename);
fprintf('Output directory: %s\n', subj_outpath);
fprintf('Total .set files: 21 (1 A + 4 B + 4 C1 avgref + 8 C2/C3 bipolar + 4 C4 c1ref)\n');
fprintf('Total .mat files: 21 (in dl_export/)\n');

end

%% ========================================================================
%  LOCAL HELPER: channel mapping from raw labels to canonical EGG1-EGG19
%  ========================================================================
function [mapping, omitted_chans, c1_idx_raw] = map_channels_to_canonical(raw_labels, canonical_raw_names)
% Maps raw BrainVision channel labels to canonical EGG1-EGG19 order
%
% Input:
%   raw_labels: cell array of raw channel labels from loaded data
%   canonical_raw_names: cell array of 19 required raw names in canonical order
%
% Output:
%   mapping: indices into raw_labels that correspond to canonical order
%           (NaN for missing channels)
%   omitted_chans: cell array of raw labels not in canonical list
%   c1_idx_raw: index of 'C1' in raw_labels (for C4 reference track)

    mapping = nan(size(canonical_raw_names));
    omitted_chans = {};
    c1_idx_raw = nan;

    % Find C1 index for later use
    c1_idx = find(strcmp(raw_labels, 'C1'), 1);
    if ~isempty(c1_idx)
        c1_idx_raw = c1_idx;
    end

    % Map each canonical raw name to its index in raw data
    for i = 1:length(canonical_raw_names)
        idx = find(strcmp(raw_labels, canonical_raw_names{i}), 1);
        if ~isempty(idx)
            % Check for duplicate
            if any(mapping(1:i-1) == idx)
                error('map_channels_to_canonical:DuplicateRawChannel', ...
                    'Duplicate raw channel "%s" would be mapped twice.', ...
                    canonical_raw_names{i});
            end
            mapping(i) = idx;
        end
    end

    % Identify omitted (non-canonical) channels
    mapped_indices = mapping(~isnan(mapping));
    for i = 1:length(raw_labels)
        if ~any(mapped_indices == i)
            omitted_chans{end+1} = raw_labels{i}; %#ok<AGROW>
        end
    end

end

%% ========================================================================
%  LOCAL HELPER: determine trigger strings present in the data
%  ========================================================================
function trig_types = get_trigger_strings(EEG)
    events = {EEG.event.type};
    trig_types = {};

    for trig = [1 2 4]
        candidates = {num2str(trig), ['S ' num2str(trig)], ['S  ' num2str(trig)], ...
                      ['S' num2str(trig)], ['s' num2str(trig)], ...
                      ['s ' num2str(trig)], ['s  ' num2str(trig)]};
        for c = 1:length(candidates)
            if any(strcmp(events, candidates{c}))
                trig_types{end+1} = candidates{c}; %#ok<AGROW>
                break;
            end
        end
    end

    if isempty(trig_types)
        warning('No matching triggers found for codes 1, 2, 4.');
        for i = 1:length(events)
            val = str2double(events{i});
            if ~isnan(val) && ismember(val, [1 2 4])
                trig_types{end+1} = events{i}; %#ok<AGROW>
            end
        end
        trig_types = unique(trig_types);
    end
end

%% ========================================================================
%  LOCAL HELPER: collect across-subject stats per track and channel
%  ========================================================================
function update_subject_norm_stats(EEG, track_name, stats_file)
% Collect mean/std ingredients per track and per channel label across subjects.
% Pool across time and epochs within each channel.

    if exist(stats_file, 'file')
        S = load(stats_file, 'stats');
        stats = S.stats;
    else
        stats = struct();
    end

    if ~isfield(stats, track_name)
        stats.(track_name) = struct();
    end

    for ch = 1:EEG.nbchan
        ch_label = EEG.chanlocs(ch).labels;
        x = double(EEG.data(ch, :, :));
        x = x(:);
        x = x(isfinite(x));

        if isempty(x)
            warning('update_subject_norm_stats:EmptyChannel', ...
                'Track %s, channel %s has no finite data. Skipping.', track_name, ch_label);
            continue;
        end

        if ~isfield(stats.(track_name), ch_label)
            stats.(track_name).(ch_label) = struct('n', 0, 'sum', 0, 'sumsq', 0);
        end

        stats.(track_name).(ch_label).n     = stats.(track_name).(ch_label).n     + numel(x);
        stats.(track_name).(ch_label).sum   = stats.(track_name).(ch_label).sum   + sum(x);
        stats.(track_name).(ch_label).sumsq = stats.(track_name).(ch_label).sumsq + sum(x.^2);
    end

    save(stats_file, 'stats');
end

%% ========================================================================
%  LOCAL HELPER: apply across-subject normalization per track and channel
%  ========================================================================
function EEG = apply_subject_norm(EEG, track_name, stats_file)

    if ~exist(stats_file, 'file')
        error('apply_subject_norm:MissingStats', ...
            ['Stats file not found: %s\n' ...
             'Run all subjects first with ZSCORE_MODE = ''collect'', ' ...
             'then rerun with ZSCORE_MODE = ''apply''.'], ...
             stats_file);
    end

    S = load(stats_file, 'stats');
    stats = S.stats;

    if ~isfield(stats, track_name)
        error('apply_subject_norm:MissingTrackStats', ...
            'No stats found for track %s in %s', track_name, stats_file);
    end

    original_class = class(EEG.data);
    data = double(EEG.data);

    zinfo = struct();
    zinfo.method = 'across_subject_per_track_per_channel';
    zinfo.track_name = track_name;
    zinfo.stats_file = stats_file;
    zinfo.channels = struct();

    for ch = 1:EEG.nbchan
        ch_label = EEG.chanlocs(ch).labels;

        if ~isfield(stats.(track_name), ch_label)
            warning('apply_subject_norm:MissingChannelStats', ...
                'Missing stats for track %s, channel %s. Leaving channel unchanged.', ...
                track_name, ch_label);
            continue;
        end

        st = stats.(track_name).(ch_label);

        if st.n < 2
            warning('apply_subject_norm:TooFewSamples', ...
                'Too few samples for track %s, channel %s. Leaving channel unchanged.', ...
                track_name, ch_label);
            continue;
        end

        mu = st.sum / st.n;
        varx = (st.sumsq - (st.sum^2 / st.n)) / max(st.n - 1, 1);
        sigma = sqrt(max(varx, eps));

        data(ch,:,:) = (data(ch,:,:) - mu) ./ sigma;

        zinfo.channels.(ch_label) = struct('mu', mu, 'sigma', sigma, 'n', st.n);
    end

    EEG.data = cast(data, original_class);
    EEG.etc.zscore = zinfo;
end
