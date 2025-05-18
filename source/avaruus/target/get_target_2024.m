%--------------------------------------------------------------------------
% Table of Contents:
% 1. Initialization and Settings
% 2. Generate Index Grid
% 3. List and Sort Data Files
% 4. Preallocate Variables
% 5. Load and Extract Data
%   5.1 Magnetic Field B_e
%   5.2 Magnetic Field B_n
%   5.3 Magnetic Field B_u
% 6. Save Extracted Data
%--------------------------------------------------------------------------

%% 1. Initialization and Settings
% Clear workspace and command window
clear; clc;

% User-defined parameters
datapath   = '/nfs/revontuli/data/akv020/secs/2024';   % Base folder for SECs data
resultpath = '/nfs/revontuli/data/akv020/secs/target'; % Output folder

% Target window settings:
% (x,y) = center pixel; W/H = half-window size in X/Y directions
xc = 11; yc = 11;    % center indices
W  = 1; H  = 1;      % half-window widths

%% 2. Generate Index Grid
% Define index ranges for submatrix extraction
dx = (xc - W + 1) : (xc + W - 1);
dy = (yc - H + 1) : (yc + H - 1);

%% 3. List and Sort Data Files
% Find all .csv files in subfolders Be, Bn, Bu and sort by name (chronological)
Be_files = dir(fullfile(datapath, 'Be', '*.csv'));
Be_files = sort({Be_files.name});
Bn_files = dir(fullfile(datapath, 'Bn', '*.csv'));
Bn_files = sort({Bn_files.name});
Bu_files = dir(fullfile(datapath, 'Bu', '*.csv'));
Bu_files = sort({Bu_files.name});

nBe = numel(Be_files);
nBn = numel(Bn_files);
nBu = numel(Bu_files);

%% 4. Preallocate Variables
% Preallocate arrays for speed
times = datetime.empty(nBe,0);  % assuming matching lengths
Be_vals = zeros(nBe, numel(dx)*numel(dy));
Bn_vals = zeros(nBn, numel(dx)*numel(dy));
Bu_vals = zeros(nBu, numel(dx)*numel(dy));

%% 5. Load and Extract Data
% Loop over Be, Bn, Bu files to read and extract submatrix values
% Recommend using readmatrix for faster raw CSV reading

% 5.1 Magnetic Field B_e
fprintf('Loading Be data...\n');
for i = 1:nBe
    fname = Be_files{i};
    fullf = fullfile(datapath, 'Be', fname);
    % Read numeric matrix
    M = readmatrix(fullf);
    % Extract subwindow and reshape to row vector
    Be_vals(i, :) = M(dx, dy);
    % Extract datetime from filename: 'Be_YYYYMMDD_HHMMSS.csv'
    dt_str = extractBetween(fname, 'Be_', '.csv');
    times(i,1) = datetime(dt_str, 'InputFormat', 'yyyyMMdd_HHmmss');
    % Progress display
    if mod(i, ceil(nBe/10)) == 0
        fprintf(' %d%% complete\n', round(100*i/nBe));
    end
end

% 5.2 Magnetic Field B_n
fprintf('Loading Bn data...\n');
for i = 1:nBn
    fname = Bn_files{i};
    fullf = fullfile(datapath, 'Bn', fname);
    M = readmatrix(fullf);
    Bn_vals(i, :) = M(dx, dy);
end

% 5.3 Magnetic Field B_u
fprintf('Loading Bu data...\n');
for i = 1:nBu
    fname = Bu_files{i};
    fullf = fullfile(datapath, 'Bu', fname);
    M = readmatrix(fullf);
    Bu_vals(i, :) = M(dx, dy);
end

%% 6. Save Extracted Data
% Combine into table and write to CSV
T = table(times, Be_vals, Bn_vals, Bu_vals, ...
    'VariableNames', {'DateTime', 'Be', 'Bn', 'Bu'});

% Ensure output directory exists
if ~exist(resultpath, 'dir')
    mkdir(resultpath);
end

outFile = fullfile(resultpath, 'extracted_secs_data_2024.csv');
writetable(T, outFile);
fprintf('Data saved to %s\n', outFile);
