function export_for_dl(EEG_in, track_name, output_path)
% export_for_dl – Export EEGLAB dataset to .mat for deep learning
%
% UPDATED:
% - Robust trigger parsing for BrainVision/EEGLAB event labels like:
%     s1, s2, s4, s128
%     Stimulus/s1, stimulus/s2, etc.
%     1, 2, 4, 128
%     S  1, S 2, s1
%
% Usage:
%   export_for_dl(EEG_in, track_name, output_path)
%
% Saves a .mat (v7.3) with variables:
%   data   – channels x timepoints x trials (or channels x timepoints for continuous)
%   labels – numeric trial labels (e.g. 1, 2, 4, 128; NaN if unknown)
%   meta   – struct with srate, channels, times, track, epoch_qc, channel mapping info,
%            and original event label info

data = EEG_in.data;  % channels x timepoints [x trials]

labels = nan(EEG_in.trials, 1);
original_eventtype = strings(EEG_in.trials, 1);

for ep = 1:EEG_in.trials
    ev = EEG_in.epoch(ep).eventtype;

    % If multiple event types are attached to the epoch, try to find the condition trigger.
    if iscell(ev)
        ev_list = string(ev(:));
    else
        ev_list = string(ev);
    end

    parsed_label = NaN;
    chosen_raw = "";

    % Try every attached event until we find a recognizable trigger
    for ii = 1:numel(ev_list)
        raw_str = ev_list(ii);
        label_num = parse_trigger_label(raw_str);

        if ~isnan(label_num)
            parsed_label = label_num;
            chosen_raw = raw_str;
            break;
        end
    end

    % If none matched, fall back to first event for bookkeeping
    if chosen_raw == "" && ~isempty(ev_list)
        chosen_raw = ev_list(1);
    end

    labels(ep) = parsed_label;
    original_eventtype(ep) = chosen_raw;

    if isnan(parsed_label)
        warning('export_for_dl:UnknownEventType', ...
            'Could not parse epoch %d event type for track %s. Raw event(s): %s', ...
            ep, track_name, strjoin(cellstr(ev_list), ', '));
    end
end

meta = struct();
meta.srate    = EEG_in.srate;
meta.channels = {EEG_in.chanlocs.labels};
meta.times    = EEG_in.times;
meta.track    = track_name;

% Save original event labels and parsed labels for debugging
meta.original_eventtype = original_eventtype;
meta.parsed_label = labels;

if isfield(EEG_in.etc, 'epoch_qc')
    meta.epoch_qc = EEG_in.etc.epoch_qc;
end

% Include channel mapping information if available
if isfield(EEG_in.etc, 'channel_mapping')
    meta.channel_mapping = EEG_in.etc.channel_mapping;
end

outfile = fullfile(output_path, [track_name '.mat']);
save(outfile, 'data', 'labels', 'meta', '-v7.3');

fprintf('  Exported %s: [%d x %d x %d] -> %s\n', ...
    track_name, size(data,1), size(data,2), size(data,3), outfile);

u = unique(labels(~isnan(labels)));
fprintf('  Parsed labels found: ');
disp(u(:)');

n_nan = sum(isnan(labels));
if n_nan > 0
    fprintf('  Warning: %d epochs had unparsed labels (saved as NaN).\n', n_nan);
end

end


function label_num = parse_trigger_label(raw_event)
% Parse BrainVision/EEGLAB event labels into numeric trigger codes.
%
% Examples handled:
%   's1' -> 1
%   's2' -> 2
%   's4' -> 4
%   's128' -> 128
%   'Stimulus/s1' -> 1
%   'stimulus/s4' -> 4
%   '1' -> 1
%   'S  1' -> 1
%   'S 2' -> 2

    label_num = NaN;

    s = lower(strtrim(string(raw_event)));

    % Remove spaces, so "S  1" becomes "s1"
    s = regexprep(s, '\s+', '');

    % Handle common forms explicitly
    if s == "s1" || s == "stimulus/s1" || s == "1"
        label_num = 1;
        return;
    elseif s == "s2" || s == "stimulus/s2" || s == "2"
        label_num = 2;
        return;
    elseif s == "s4" || s == "stimulus/s4" || s == "4"
        label_num = 4;
        return;
    elseif s == "s128" || s == "stimulus/s128" || s == "128"
        label_num = 128;
        return;
    end

    % Generic fallback:
    % match final s<number> or just number
    tok = regexp(char(s), '(?:stimulus/)?s?(\d+)$', 'tokens', 'once');
    if ~isempty(tok)
        label_num = str2double(tok{1});
    end
end