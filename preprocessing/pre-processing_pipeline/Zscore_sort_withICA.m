function Zscore_sort_withICA(in_dir, out_dir, fs, trial_len_sec, trigger_codes, zscore_mode)
% Zscore_sort_withICA
% Load already-epoched subject files, apply cross-subject scaling, split
% trials by trigger code, compute or collect trial quality measures, and
% save concatenated outputs for later model training.
%
% Expected input per subject file:
%   data   : channels x time x trials
%   labels : numeric trial labels such as 1, 2, or 4
%   meta   : struct containing at least channel information, and optionally
%            sampling rate and precomputed epoch quality values
%
% Saved outputs in out_dir:
%   concat_sX.mat   : concatenated data as time x channel x trial
%   origin_sX.mat   : subject IDs and per-trial origin metadata
%   quality_sX.mat  : trial quality values and column names
%
% zscore_mode:
%   "A" = scale each subject by its overall signal magnitude
%   "C" = standardize each channel using global mean and standard deviation
%
% This function assumes the data has already been epoched. It does not
% extract events from continuous recordings.

    if nargin < 6 || isempty(zscore_mode)
        zscore_mode = "A";
    end
    if nargin < 5 || isempty(trigger_codes)
        trigger_codes = [1 2 4];
    end
    if nargin < 4 || isempty(trial_len_sec)
        trial_len_sec = 16;
    end
    if nargin < 3 || isempty(fs)
        fs = 512;
    end

    % Convert trigger inputs into a numeric row vector.
    trigger_codes = normalize_trigger_codes(trigger_codes);

    if ~isfolder(in_dir)
        error("in_dir does not exist: %s", in_dir);
    end
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end

    files = dir(fullfile(in_dir, "*_B3_c1ref.mat"));
    if isempty(files)
        error("No files matching *_B3_c1ref.mat found in: %s", in_dir);
    end

    trial_len_samples_expected = round(trial_len_sec * fs);

    fprintf("=== Zscore_sort ===\n");
    fprintf("Input dir: %s\n", in_dir);
    fprintf("Output dir: %s\n", out_dir);
    fprintf("Files found: %d\n", numel(files));
    fprintf("Expected fs=%g Hz, expected trial_len=%.3f sec (%d samples)\n", ...
        fs, trial_len_sec, trial_len_samples_expected);
    fprintf("Triggers: %s\n", mat2str(trigger_codes));
    fprintf("zscore_mode=%s\n", string(zscore_mode));
    fprintf("Mode: already-epoched export_for_dl input\n\n");

    % In mode C, first estimate one global mean and standard deviation per
    % channel across all files.
    use_mode_c = strcmpi(string(zscore_mode), "C");
    global_mu = [];
    global_sigma = [];
    if use_mode_c
        fprintf("Computing global mean/std per channel (streaming)...\n");
        [global_mu, global_sigma] = compute_global_channel_stats(files);
        fprintf("Global stats computed.\n\n");
    end

    % Create one set of output containers for each trigger code.
    n_trig = numel(trigger_codes);
    concat_cells = cell(n_trig, 1);
    origin_cells = cell(n_trig, 1);
    origin_meta_cells = cell(n_trig, 1);
    quality_cells = cell(n_trig, 1);

    quality_colnames = { ...
        'amplitude_range_mean', ...
        'amplitude_range_max', ...
        'rms_mean', ...
        'rms_max', ...
        'kurtosis_mean', ...
        'kurtosis_max', ...
        'spectral_flatness_mean', ...
        'spectral_flatness_min'};

    for ti = 1:n_trig
        concat_cells{ti} = [];
        origin_cells{ti} = [];
        origin_meta_cells{ti} = struct( ...
            "participant_id", {}, ...
            "trial_index", {}, ...
            "original_trial_index_within_subject", {}, ...
            "trigger_code", {}, ...
            "event_sample_index", {}, ...
            "event_timestamp_sec", {});
        quality_cells{ti} = zeros(0, numel(quality_colnames));
    end

    % Track whether channel names and channel order stay identical across
    % all subjects.
    reference_ch_names = [];
    channel_order_ok = true;
    channel_mismatch_subjects = {};

    % Process each subject file one by one.
    for f = 1:numel(files)
        file = files(f);
        in_path = fullfile(file.folder, file.name);
        subj_id = parse_subject_id(file.name);

        S = load(in_path);

        if ~isfield(S, "data")
            error("File %s missing required field 'data'", in_path);
        end
        if ~isfield(S, "labels")
            error("File %s missing required field 'labels'", in_path);
        end
        if ~isfield(S, "meta")
            error("File %s missing required field 'meta'", in_path);
        end
        if ~isfield(S.meta, "channels")
            error("File %s missing required field 'meta.channels'", in_path);
        end

        % Use the file's own sampling rate when available, otherwise fall
        % back to the input value.
        if isfield(S.meta, "srate")
            fs_file = double(S.meta.srate);
        else
            warning("File %s missing meta.srate. Using input fs=%g", file.name, fs);
            fs_file = fs;
        end

        trial_len_samples_file_expected = round(trial_len_sec * fs_file);

        % Read and normalize channel names so they can be compared across
        % files.
        ch_names_this = normalize_ch_names(S.meta.channels);

        % Input is expected as channels x time x trials.
        data = double(S.data);
        if ndims(data) ~= 3
            error("File %s: expected 3D data (channels x time x trials), got size %s", ...
                in_path, mat2str(size(data)));
        end

        if size(data,1) ~= numel(ch_names_this)
            error("File %s: channel count mismatch between data (%d) and meta.channels (%d)", ...
                in_path, size(data,1), numel(ch_names_this));
        end

        % Reorder to time x channel x trial, which is the internal format
        % used in this function.
        data = permute(data, [2 1 3]);

        % Compare channel names and order against the first subject file.
        if isempty(reference_ch_names)
            reference_ch_names = ch_names_this;
        else
            if ~isequal(reference_ch_names, ch_names_this)
                channel_order_ok = false;
                channel_mismatch_subjects{end+1} = subj_id; %#ok<AGROW>
                warning("Channel order/name mismatch in subject %s", subj_id);
            end
        end

        [n_time, n_ch, n_trials] = size(data);

        if numel(S.labels) ~= n_trials
            error("File %s: labels length (%d) does not match number of trials (%d)", ...
                in_path, numel(S.labels), n_trials);
        end

        if n_time ~= trial_len_samples_file_expected
            warning("Subject %s (%s): epoch length is %d samples, expected %d from %.3f sec @ %.3f Hz", ...
                subj_id, file.name, n_time, trial_len_samples_file_expected, trial_len_sec, fs_file);
        end

        labels = double(S.labels(:));

        % Apply either global channel standardization or subject-level
        % scaling, depending on the chosen mode.
        if use_mode_c
            data = standardize_global_3d(data, global_mu, global_sigma);
        else
            data = scale_subject_3d(data);
        end

        % Use stored epoch quality values if they are available and shaped
        % correctly. Otherwise compute them from the scaled data.
        quality_this = [];

        if isfield(S.meta, "epoch_qc")
            q_candidate = S.meta.epoch_qc;

            if isnumeric(q_candidate) && size(q_candidate,1) == n_trials && size(q_candidate,2) == numel(quality_colnames)
                quality_this = double(q_candidate);
            end
        end

        if isempty(quality_this)
            quality_this = zeros(n_trials, numel(quality_colnames));
            for tr = 1:n_trials
                epoch = data(:,:,tr);
                quality_this(tr, :) = compute_epoch_quality(epoch, fs_file);
            end
        end

        % Send each trial into the correct trigger-specific output group.
        for ti = 1:n_trig
            trig = trigger_codes(ti);
            idx = find(labels == trig);

            if isempty(idx)
                continue;
            end

            for k = 1:numel(idx)
                tr = idx(k);
                epoch = data(:,:,tr);
                qrow = quality_this(tr, :);

                if isempty(concat_cells{ti})
                    concat_cells{ti} = reshape(epoch, [size(epoch,1), size(epoch,2), 1]);
                else
                    concat_cells{ti} = cat(3, concat_cells{ti}, epoch);
                end

                quality_cells{ti}(end+1, :) = qrow; %#ok<AGROW>

                trial_index_global = size(concat_cells{ti}, 3);

                origin_entry = struct( ...
                    "participant_id", subj_id, ...
                    "trial_index", trial_index_global, ...
                    "original_trial_index_within_subject", tr, ...
                    "trigger_code", trig, ...
                    "event_sample_index", NaN, ...
                    "event_timestamp_sec", NaN ...
                );

                origin_cells{ti}(end+1, 1) = str2double(subj_id); %#ok<AGROW>
                origin_meta_cells{ti}(end+1) = origin_entry; %#ok<AGROW>
            end
        end

        fprintf("Processed subject %s (%s): [%d time x %d ch x %d trials]\n", ...
            subj_id, file.name, n_time, n_ch, n_trials);
    end

    % Check that output sizes agree and that the saved tensors contain no
    % NaN or Inf values.
    fprintf("\n=== FINAL SANITY CHECK ===\n");
    fprintf("Channel order consistent across subjects: %s\n", yesno(channel_order_ok));
    if ~channel_order_ok
        fprintf("Subjects with channel mismatch: %s\n", strjoin(channel_mismatch_subjects, ", "));
    end

    sanity_ok = channel_order_ok;

    for ti = 1:n_trig
        trig = trigger_codes(ti);
        concat = concat_cells{ti};
        origin = origin_cells{ti};
        origin_meta = origin_meta_cells{ti};
        quality = quality_cells{ti};

        if isempty(concat)
            n_time = 0;
            n_ch = 0;
            n_trials = 0;
            has_nan = false;
            has_inf = false;
        else
            [n_time, n_ch, n_trials] = size(concat);
            has_nan = any(isnan(concat(:)));
            has_inf = any(isinf(concat(:)));
        end

        origin_n = numel(origin);
        origin_meta_n = numel(origin_meta);
        quality_n = size(quality, 1);

        count_ok = (n_trials == origin_n) && (n_trials == origin_meta_n) && (n_trials == quality_n);
        content_ok = ~has_nan && ~has_inf;

        fprintf("\nTrigger s%d\n", trig);
        fprintf("  concat shape: [%d x %d x %d]\n", n_time, n_ch, n_trials);
        fprintf("  origin rows: %d\n", origin_n);
        fprintf("  origin_meta rows: %d\n", origin_meta_n);
        fprintf("  quality rows: %d\n", quality_n);
        fprintf("  counts match: %s\n", yesno(count_ok));
        fprintf("  no NaN/Inf: %s\n", yesno(content_ok));

        if ~count_ok || ~content_ok
            sanity_ok = false;
        end
    end

    if ~isempty(reference_ch_names)
        fprintf("\nReference channel order:\n");
        disp(reference_ch_names(:));
    end

    fprintf("\nOverall sanity status: %s\n", yesno(sanity_ok));

    user_reply = input('Proceed with saving? [y/n]: ', 's');
    if ~strcmpi(strtrim(user_reply), 'y')
        error("Aborted by user before saving outputs.");
    end

    % Save one concat, origin, and quality file for each trigger code.
    for ti = 1:n_trig
        trig = trigger_codes(ti);

        concat = concat_cells{ti}; %#ok<NASGU>
        if isempty(concat)
            warning("No trials found for trigger s%d. Saving empty outputs.", trig);
            concat = zeros(0, 0, 0);
        end

        origin = origin_cells{ti}; %#ok<NASGU>
        origin_meta = origin_meta_cells{ti}; %#ok<NASGU>
        quality = quality_cells{ti}; %#ok<NASGU>
        quality_columns = quality_colnames; %#ok<NASGU>

        concat_name  = fullfile(out_dir, sprintf("concat_s%d.mat", trig));
        origin_name  = fullfile(out_dir, sprintf("origin_s%d.mat", trig));
        quality_name = fullfile(out_dir, sprintf("quality_s%d.mat", trig));

        save(concat_name, "concat", "-v7.3");
        save(origin_name, "origin", "origin_meta", "-v7.3");
        save(quality_name, "quality", "quality_columns", "-v7.3");

        fprintf("Saved trigger s%d: concat shape = [%d x %d x %d]\n", ...
            trig, size(concat,1), size(concat,2), size(concat,3));
        fprintf("  %s\n  %s\n  %s\n", concat_name, origin_name, quality_name);
    end

    fprintf("\nDone.\n");
end

function trig_out = normalize_trigger_codes(trigger_codes)
% Convert trigger labels into a numeric row vector. Accepts either numeric
% input or strings such as "1", "s1", or similar forms.

    if isnumeric(trigger_codes)
        trig_out = double(trigger_codes(:))';
        return;
    end

    trigger_codes = string(trigger_codes(:));
    trig_out = zeros(1, numel(trigger_codes));

    for i = 1:numel(trigger_codes)
        s = lower(strtrim(trigger_codes(i)));
        s = regexprep(s, '\s+', '');

        tok = regexp(s, 's?(\d+)$', 'tokens', 'once');
        if isempty(tok)
            error("Could not parse trigger code from '%s'", trigger_codes(i));
        end
        trig_out(i) = str2double(tok{1});
    end
end

function subj_id = parse_subject_id(filename)
% Extract the subject ID from filenames that end in _B3_c1ref.mat. If the
% pattern is not found, fall back to the filename without extension.

    expr = "(\d+)_B3_c1ref\.mat";
    tok = regexp(filename, expr, "tokens", "once");
    if isempty(tok)
        [~, base, ~] = fileparts(filename);
        subj_id = base;
        return;
    end
    subj_id = tok{1};
end

function out = normalize_ch_names(ch_names)
% Convert channel names into a clean column cell array of trimmed strings.

    if isstring(ch_names)
        out = cellstr(ch_names(:));
    elseif iscell(ch_names)
        out = cell(size(ch_names));
        for i = 1:numel(ch_names)
            if isstring(ch_names{i})
                out{i} = char(ch_names{i});
            else
                out{i} = char(string(ch_names{i}));
            end
        end
        out = out(:);
    else
        out = cellstr(string(ch_names(:)));
    end

    for i = 1:numel(out)
        out{i} = strtrim(out{i});
    end
end

function [mu, sigma] = compute_global_channel_stats(files)
% Compute a global mean and standard deviation for each channel across all
% subjects, time points, and trials without loading everything into memory
% at once.

    mu = [];
    M2 = [];
    n_total = 0;

    for f = 1:numel(files)
        in_path = fullfile(files(f).folder, files(f).name);
        S = load(in_path);

        if ~isfield(S, "data")
            error("Missing 'data' in %s", in_path);
        end
        if ~isfield(S, "meta") || ~isfield(S.meta, "channels")
            error("Missing 'meta.channels' in %s", in_path);
        end

        ch_names = normalize_ch_names(S.meta.channels);
        X = double(S.data);

        if ndims(X) ~= 3
            error("Expected 3D data in %s, got size %s", in_path, mat2str(size(X)));
        end

        if size(X,1) ~= numel(ch_names)
            error("Channel count mismatch in %s", in_path);
        end

        % Flatten all time points and trials into one sample dimension while
        % keeping channels separate.
        X = permute(X, [2 1 3]);
        X = reshape(X, [], size(X,2));

        if isempty(mu)
            n_ch = size(X,2);
            mu = zeros(1, n_ch);
            M2 = zeros(1, n_ch);
        else
            if size(X,2) ~= numel(mu)
                error("Channel count mismatch in %s (got %d, expected %d)", ...
                    in_path, size(X,2), numel(mu));
            end
        end

        n_file = size(X,1);
        file_mean = mean(X, 1);
        file_var  = var(X, 0, 1);

        if n_total == 0
            mu = file_mean;
            M2 = file_var * max(n_file - 1, 1);
            n_total = n_file;
        else
            delta = file_mean - mu;
            n_new = n_total + n_file;

            mu = mu + delta * (n_file / n_new);
            M2 = M2 + (file_var * max(n_file - 1, 1)) + (delta.^2) * (n_total * n_file / n_new);
            n_total = n_new;
        end
    end

    if n_total < 2
        error("Not enough samples globally to compute std.");
    end

    sigma = sqrt(M2 / (n_total - 1));
    sigma(sigma == 0) = 1;
end

function Xz = standardize_global_3d(X, mu, sigma)
% Standardize a time x channel x trial tensor using global per-channel
% statistics.

    if isempty(mu) || isempty(sigma)
        error("Global mu/sigma are empty. Did you compute them?");
    end

    if size(X,2) ~= numel(mu)
        error("Channel count mismatch in standardize_global_3d.");
    end

    [n_time, n_ch, n_trials] = size(X);
    X2 = reshape(X, [], n_ch);
    X2 = (X2 - mu) ./ sigma;
    Xz = reshape(X2, n_time, n_ch, n_trials);
end

function Xs = scale_subject_3d(X)
% Scale one subject's full tensor by the square root of the mean channel
% variance across all time points and trials.

    [~, n_ch, ~] = size(X);
    X2 = reshape(X, [], n_ch);

    mean_var = mean(var(X2, 0, 1));
    scale = sqrt(mean_var);

    if scale == 0 || ~isfinite(scale)
        warning("Subject has non-finite scaling (mean_var=%.6g). Leaving data unchanged.", mean_var);
        Xs = X;
    else
        Xs = X ./ scale;
    end
end

function qrow = compute_epoch_quality(epoch, fs)
% Compute a small set of per-epoch quality measures across channels.

    epoch = double(epoch);

    amp_range = max(epoch, [], 1) - min(epoch, [], 1);
    rms_ch = sqrt(mean(epoch.^2, 1));
    kurt_ch = kurtosis(epoch, 0, 1);

    % Spectral flatness is estimated separately for each channel.
    sf_ch = zeros(1, size(epoch,2));
    for ch = 1:size(epoch,2)
        x = epoch(:, ch);
        [Pxx, ~] = pwelch(x, [], [], [], fs);

        Pxx = Pxx + eps;
        gmean = exp(mean(log(Pxx)));
        amean = mean(Pxx);
        sf_ch(ch) = gmean / amean;
    end

    qrow = [ ...
        mean(amp_range), ...
        max(amp_range), ...
        mean(rms_ch), ...
        max(rms_ch), ...
        mean(kurt_ch), ...
        max(kurt_ch), ...
        mean(sf_ch), ...
        min(sf_ch)];
end

function s = yesno(tf)
% Convert a logical value into YES or NO for printed summaries.

    if tf
        s = 'YES';
    else
        s = 'NO';
    end
end