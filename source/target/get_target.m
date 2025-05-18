% This script gets target data from secs maps

%% Define workpath
workpath = '/Users/akv020/Tensorflow/fennomag-net/data/secs/test_mode_2024';

%% Define target dimension
x = 11; % midpoint
y = 11; % midpoint
W = 1; % please define correctly dimension 
H = 1; % please define correctly dimension 
dx = (x-W+1:x+W-1); % get relevant coordinates 
dy = (x-H+1:x+H-1); % get relevant coordinates 

%% Get Be names
cd(workpath)
Be_names = dir('Be/*.csv');

%% Get Be data 
Be_length = numel(Be_names);
etick = 1;
for iE = 1:Be_length

    if iE > etick
        disp(iE)
        etick = etick + Be_length/100;
    end
    % read table and convert to array
    Be_data = table2array(readtable(fullfile(['Be/',Be_names(iE).name])));
    Be(iE) = Be_data(dx,dy);
end

%% Get Bn names 
Bn_names = dir('Bn/*.csv');

%% Get Bn data 
for iN = 1:numel(Bn_names)
    % read table and convert to array
    Bn_data = table2array(readtable(fullfile(['Bn/',Bn_names(iN).name])));
    Bn(iN) = Bn_data(dx,dy);
end

%% Get Bu names 
Bu_names = dir('Bu/*.csv');

%% Get Bu data 
for iU = 1:numel(Bu_names)
    % read table and convert to array
    Bu_data = table2array(readtable(fullfile(['Bu/',Bu_names(iU).name])));
    Bu(iU) = Bu_data(dx,dy);
end