clearvars; close all; clc
%% Multi-track EGG cleaning pipeline — batch caller
%
% Loops over all .vhdr files in the raw data folder and runs the
% egg_clean_multitrack pipeline on each one.  Produces per-subject output
% folders with:
%   - Track A  (broadband, minimal clean)
%   - Track B1 (gastric slow wave, 0.03–0.15 Hz)
%   - Track B2 (intestinal slow wave, 0.15–0.5 Hz)
%   - Track B3 (neural burst band, 1–200 Hz)
%   - Track B4 (autonomic modulation, 0.5–1 Hz)
%   - Track C  (average ref + bipolar variants of each B-track)
%   - dl_export/ subfolder with .mat files for deep learning
%
% Requires: EEGLAB (with bva-io and cleanline plugins) on the MATLAB path.

%% ---- PATHS (edit these) ------------------------------------------------
datapath = '/home/gutproject/Desktop/guteeg/gut-eeg/data/gut_renamed';
outpath  = '/home/gutproject/Desktop/guteeg/gut-eeg/data/gut_tensors_ICA';

% Add EEGLAB to path if not already present
eeglabdir = '/home/gutproject/Documents/MATLAB/eeglab/';
if ~exist('eeglab', 'file')
    addpath(eeglabdir);
end

%% ---- COLLECT FILES -----------------------------------------------------
files = dir(fullfile(datapath, '*.vhdr'));

if isempty(files)
    fprintf('No .vhdr files found in: %s\n', datapath);
    return;
end

fprintf('Found %d .vhdr files.\n', length(files));

%% ---- CONTAINERS FOR LOGGING -------------------------------------------
successFiles = {};
failedFiles  = {};
failedMsgs   = {};

%% ---- MAIN LOOP ---------------------------------------------------------
for k = 1:length(files)

    filename = files(k).name;
    fprintf('\n[%d/%d] Processing: %s\n', k, length(files), filename);

    try
        egg_clean_multitrack_UPDATED(datapath, outpath, filename);
        fprintf('   -> OK: %s\n', filename);
        successFiles{end+1} = filename; %#ok<SAGROW>
    catch ME
        warning('   -> ERROR in %s: %s', filename, ME.message);
        failedFiles{end+1} = filename;   %#ok<SAGROW>
        failedMsgs{end+1}  = ME.message; %#ok<SAGROW>
    end

end

%% ---- SUMMARY -----------------------------------------------------------
fprintf('\n========== Batch finished ==========\n');
fprintf('  Successful: %d\n', numel(successFiles));
fprintf('  Failed:     %d\n', numel(failedFiles));

if ~isempty(failedFiles)
    fprintf('\nFailed files:\n');
    for i = 1:numel(failedFiles)
        fprintf('  %s  -->  %s\n', failedFiles{i}, failedMsgs{i});
    end
end
