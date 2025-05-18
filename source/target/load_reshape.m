% Table of Contents
% 1. Load Data
% 2. Combine Data
% 3. Resample Data to 15-minute Trailing Averages (Explicit Loop with Progress)
% 4. Visualize Components
% 5. Save Resampled Data

%% 1. Load Data
% Prompt user for the folder containing the yearly CSV files
dataFolder = '/Users/akv020/Tensorflow/fennomag-net/data/target';
filePattern = fullfile(dataFolder, 'extracted_secs_data_*.csv');
csvFiles = dir(filePattern);

yearlyTables = cell(numel(csvFiles),1);
for k = 1:numel(csvFiles)
    baseName = csvFiles(k).name;
    fullName = fullfile(dataFolder, baseName);
    fprintf('Loading %s...\n', baseName);
    tbl = readtable(fullName, 'Format', '%{dd-MMM-yyyy HH:mm:ss}D%f%f%f', ...
                    'Delimiter', ',', 'ReadVariableNames', true);
    tbl.Properties.VariableNames = {'DateTime','Be','Bn','Bu'};
    yearlyTables{k} = tbl;
end

%% 2. Combine Data
fprintf('Combining yearly data...\n');
allData = vertcat(yearlyTables{:});
% Convert to timetable, using the DateTime column as row times
tt = table2timetable(allData, 'RowTimes', 'DateTime');

%% 3. Resample Data to 15-minute Trailing Averages (Explicit Loop with Progress)
fprintf('Computing trailing 15-minute averages with progress updates...\n');

% Define start and end of full period using the original DateTime
startTime = tt.DateTime(1);
endTime   = tt.DateTime(end);

% Initialize first quarter-hour boundary
t_current = dateshift(startTime, 'start', 'minute');
if rem(minute(t_current),15) ~= 0
    t_current = dateshift(t_current, 'start', 'hour') + minutes(floor(minute(t_current)/15)*15);
end
if t_current < startTime
    t_current = t_current + minutes(15);
end

% Estimate total number of windows for progress calculation
totalMinutes = minutes(endTime - t_current);
nSteps = floor(totalMinutes/15);
stepCount = 0;
nextPrint = 0.1;  % next percent threshold to print

% Prepare arrays for results
dates15 = datetime.empty(0,1);
Be15    = [];
Bn15    = [];
Bu15    = [];

while t_current + minutes(15) <= endTime
    t_next = t_current + minutes(15);
    stepCount = stepCount + 1;

    % Print progress every 0.1%
    pct = (stepCount / nSteps) * 100;
    if pct >= nextPrint
        fprintf('Progress: %.1f%%\n', nextPrint);
        nextPrint = nextPrint + 0.1;
    end

    % Compute trailing-average window mask using DateTime    
    mask = tt.DateTime > t_current & tt.DateTime <= t_next;
    if any(mask)
        dates15(end+1,1) = t_next;
        Be15(end+1,1)    = mean(tt.Be(mask));
        Bn15(end+1,1)    = mean(tt.Bn(mask));
        Bu15(end+1,1)    = mean(tt.Bu(mask));
    end
    t_current = t_next;
end

% Build new timetable using the computed quarter-hour timestamps
TT15 = timetable(dates15, Be15, Bn15, Bu15, 'VariableNames', {'Be','Bn','Bu'});

%% 4. Visualize Components
fprintf('Plotting results...\n');
figure;
plot(TT15.dates15, TT15.Be);
datetick('x','keeplimits');
title('Trailing 15-min Average: Be'); xlabel('Time'); ylabel('Be'); grid on;

figure;
plot(TT15.dates15, TT15.Bn);
datetick('x','keeplimits');
title('Trailing 15-min Average: Bn'); xlabel('Time'); ylabel('Bn'); grid on;

figure;
plot(TT15.dates15, TT15.Bu);
datetick('x','keeplimits');
title('Trailing 15-min Average: Bu'); xlabel('Time'); ylabel('Bu'); grid on;

%% 5. Save Resampled Data
fprintf('Saving to CSV...\n');
exportTbl = timetable2table(TT15, 'ConvertRowTimes', true);
exportTbl.Be = single(exportTbl.Be);
exportTbl.Bn = single(exportTbl.Bn);
exportTbl.Bu = single(exportTbl.Bu);

outputFile = fullfile(dataFolder, 'extracted_secs_data_2010_2024_15min_trailing.csv');
writetable(exportTbl, outputFile);
fprintf('Saved file: %s\n', outputFile);
