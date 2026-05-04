in_dir  = '/home/gutproject/Desktop/guteeg/gut-eeg/data/processed_sub_withICA_v2';
out_dir = '/home/gutproject/Desktop/guteeg/gut-eeg/data/concatenated_withICA_v2';

fs = 512;
trial_len_sec = 16;
trigger_codes = ["stimulus/s1" "stimulus/s2" "stimulus/s4"];

zscore_mode = 'A';

Zscore_sort_withICA(in_dir, out_dir, fs, trial_len_sec, trigger_codes, zscore_mode);