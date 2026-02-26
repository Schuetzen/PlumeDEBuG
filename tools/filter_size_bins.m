%---------------------------------------------------------------------
% Script: proportionalBinFilter.m
%
% Description:
% This script filters the aggregated results MAT file to reduce the number of bubbles
% in user-specified diameter bins using a proportional reduction approach. The script:
% 1. Applies more aggressive filtering to bins with higher counts
% 2. Creates smoother transitions between filtered and non-filtered regions
% 3. Preserves the overall shape of the distribution while reducing peaks
%
% The script:
% 1. Creates a backup of the original aggregated_results.mat file
% 2. Loads the backup file
% 3. Applies proportional filtering to bins that exceed the threshold
% 4. Saves the filtered data as the new aggregated_results.mat
%
% Instructions:
% 1. Set the baseFolder variable to point to your data directory
% 2. Configure the targetRange array to define your diameter range to filter
% 3. Set the targetBinCount and binWidth to control filtering parameters
% 4. Run this script in MATLAB
%
% Dependencies:
% - MATLAB environment with support for basic file I/O functions
% - A properly structured directory with the aggregated results file
%
% Author:
% Schuetzen Jung
%
% Date:
% 03-05-2025
%---------------------------------------------------------------------

%% User Configuration
% Set the base folder (change this value to point to your data folder)
baseFolder = '../Aggregated_bubble_data';

% Define the overall target range for analysis (region of interest)
targetRange = [0.0017, 0.0040]; % [min_diameter, max_diameter]

% Define the bin width for analysis (used for histogram binning)
binWidth = 0.0001; % Adjust based on your data distribution

% Target maximum bin count (bins over this will be reduced)
targetBinCount = 1200;

% Number of bins for full histogram
numHistBins = 200;

% Set the reduction approach
% 1 = Linear reduction (more removal from higher count bins)
% 2 = Proportional reduction (scales with distance above threshold)
% 3 = Progressive reduction (most aggressive on highest peaks)
reductionMethod = 3;

%% Define file paths using baseFolder
origMatFile = fullfile(baseFolder, 'aggregated_results.mat');
backupMatFile = fullfile(baseFolder, 'aggregated_results_backup.mat');

%% Check if the original file exists
if ~exist(origMatFile, 'file')
    error('File %s does not exist.', origMatFile);
end

%% Check if a backup file already exists to avoid overwriting it
if exist(backupMatFile, 'file')
    warning('Backup file %s already exists. Using the existing backup.', backupMatFile);
else
    % Rename the original file to create a backup
    movefile(origMatFile, backupMatFile);
    fprintf('Renamed %s to %s\n', origMatFile, backupMatFile);
end

%% Load the backup file
S = load(backupMatFile);

%% Determine which variable contains the aggregated data
if isfield(S, 'combinedImgInfo')
    data = S.combinedImgInfo;
    varName = 'combinedImgInfo';
elseif isfield(S, 'imgInfo')
    data = S.imgInfo;
    varName = 'imgInfo';
else
    error('The file does not contain variable "combinedImgInfo" or "imgInfo".');
end

%% Get the number of records before filtering
numRecordsBefore = numel(data);

%% Extract all bubble diameters
try
    allDiameters = [data.bubble_diameter];
    fprintf('Successfully extracted %d bubble diameters\n', numel(allDiameters));
catch e
    fprintf('Error extracting bubble diameters: %s\n', e.message);
    error('Failed to extract bubble diameters. Please check the structure of your data.');
end

%% Create histograms for the full dataset
% Calculate histogram bins for the full dataset
minDiameter = min(allDiameters);
maxDiameter = max(allDiameters);
fprintf('Diameter range in full dataset: %.5f to %.5f\n', minDiameter, maxDiameter);

% Calculate edges for full histogram
fullHistEdges = linspace(minDiameter, maxDiameter, numHistBins + 1);
[fullHistCounts, ~] = histcounts(allDiameters, fullHistEdges);

% Create histogram edges based on bin width for target range
histEdges = targetRange(1):binWidth:targetRange(2);
inRangeDiameters = allDiameters(allDiameters >= targetRange(1) & allDiameters <= targetRange(2));
[histCounts, ~] = histcounts(inRangeDiameters, histEdges);

fprintf('Number of bubbles in target range %.5f - %.5f: %d (%.1f%% of total)\n', ...
    targetRange(1), targetRange(2), numel(inRangeDiameters), ...
    100 * numel(inRangeDiameters) / numel(allDiameters));

% Find bins that exceed the target count in the target range
binsToFilter = find(histCounts > targetBinCount);
binCenters = histEdges(1:end-1) + binWidth/2;

%% Plot original histogram for full dataset
figure('Name', 'Full Bubble Distribution Before Filtering', 'Position', [100, 100, 1000, 500]);
histogram('BinEdges', fullHistEdges, 'BinCounts', fullHistCounts);
title('Full Bubble Diameter Distribution (Before Filtering)');
xlabel('Bubble Diameter (m)');
ylabel('Count');
grid on;

% Highlight the target range
hold on;
xline(targetRange(1), 'r-', 'LineWidth', 1.5);
xline(targetRange(2), 'r-', 'LineWidth', 1.5);
text(targetRange(1), 0.9*max(ylim), 'Target Range Start', 'Color', 'r', 'HorizontalAlignment', 'left');
text(targetRange(2), 0.9*max(ylim), 'Target Range End', 'Color', 'r', 'HorizontalAlignment', 'right');
hold off;

% Save the axis limits for later use
fullXLim = xlim;
fullYLim = ylim;

%% Plot detailed histogram of target range
figure('Name', 'Target Range Before Filtering');
histogram('BinEdges', histEdges, 'BinCounts', histCounts);
title('Target Range Bubble Distribution (Before Filtering)');
xlabel('Bubble Diameter (m)');
ylabel('Count');
grid on;

% Highlight bins that need filtering
hold on;
for i = 1:numel(binsToFilter)
    binIdx = binsToFilter(i);
    x = [histEdges(binIdx), histEdges(binIdx+1), histEdges(binIdx+1), histEdges(binIdx)];
    y = [0, 0, histCounts(binIdx), histCounts(binIdx)];
    patch(x, y, 'red', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Add count text
    text(binCenters(binIdx), histCounts(binIdx)*1.05, sprintf('%d', histCounts(binIdx)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Show threshold line
plot([histEdges(1), histEdges(end)], [targetBinCount, targetBinCount], 'r--', 'LineWidth', 1.5);
text(histEdges(1), targetBinCount*1.05, sprintf('Target: %d', targetBinCount), ...
    'Color', 'r', 'FontWeight', 'bold');
hold off;

%% Apply proportional bin-based filtering
% Array to store indices of bubbles to keep
indicesToKeep = [];

% First, keep all bubbles outside the target range
outsideRangeIndices = find(allDiameters < targetRange(1) | allDiameters > targetRange(2));
indicesToKeep = [indicesToKeep; outsideRangeIndices(:)];

fprintf('\nStarting proportional bin filtering:\n');
fprintf('------------------------------------------\n');
fprintf('%-20s %-10s %-10s %-10s %-10s\n', 'Bin Range (m)', 'Count', 'Keep', 'Remove', 'Reduction %');
fprintf('------------------------------------------\n');

% Process each bin in the histogram
for i = 1:numel(histCounts)
    binMin = histEdges(i);
    binMax = histEdges(i+1);
    
    % Find indices of bubbles in this bin
    binIndices = find(allDiameters >= binMin & allDiameters < binMax);
    binCount = numel(binIndices);
    
    % Skip empty bins
    if binCount == 0
        continue;
    end
    
    % Determine how many bubbles to keep based on the reduction method
    if binCount <= targetBinCount
        % Keep all bubbles in this bin (under threshold)
        numToKeep = binCount;
        fprintf('%.5f - %.5f     %-10d %-10d %-10d %.1f%%\n', ...
            binMin, binMax, binCount, numToKeep, 0, 0.0);
    else
        % Bin exceeds threshold, apply reduction based on selected method
        switch reductionMethod
            case 1 % Linear reduction
                % The higher the count, the more aggressive the reduction
                reduction = min(0.8, (binCount - targetBinCount) / binCount);
                numToKeep = round(binCount * (1 - reduction));
                
            case 2 % Proportional reduction
                % Reduction proportional to how much the bin exceeds the threshold
                excess = binCount - targetBinCount;
                reduction = min(0.8, excess / binCount);
                numToKeep = binCount - round(excess * (1 - reduction/2));
                
            case 3 % Progressive reduction
                % More aggressive on highest peaks with smooth transition
                excessRatio = (binCount - targetBinCount) / targetBinCount;
                % Apply a progressive reduction factor
                if excessRatio <= 0.5
                    reduction = 0.3 * excessRatio;
                elseif excessRatio <= 1.0
                    reduction = 0.15 + 0.4 * (excessRatio - 0.5);
                else
                    reduction = 0.35 + 0.45 * min(1, (excessRatio - 1.0));
                end
                numToKeep = round(targetBinCount + (binCount - targetBinCount) * (1 - reduction));
        end
        
        % Ensure we're keeping at least the target count
        numToKeep = max(targetBinCount, numToKeep);
        
        % Report the filtering for this bin
        fprintf('%.5f - %.5f     %-10d %-10d %-10d %.1f%%\n', ...
            binMin, binMax, binCount, numToKeep, binCount - numToKeep, ...
            100 * (binCount - numToKeep) / binCount);
    end
    
    % Randomly select bubbles to keep
    if numToKeep < binCount
        keepFromBin = binIndices(randperm(binCount, numToKeep));
    else
        keepFromBin = binIndices;
    end
    
    % Add to our list of indices to keep
    indicesToKeep = [indicesToKeep; keepFromBin(:)];
end

%% Sort indices to maintain original order
indicesToKeep = sort(indicesToKeep);

%% Filter data based on the indices to keep
filteredData = data(indicesToKeep);

%% Get the number of records after filtering
numRecordsAfter = numel(filteredData);
fprintf('------------------------------------------\n');
fprintf('Total records before filtering: %d\n', numRecordsBefore);
fprintf('Total records after filtering: %d\n', numRecordsAfter);
fprintf('Reduction: %d records (%.2f%%)\n', numRecordsBefore - numRecordsAfter, ...
    100 * (numRecordsBefore - numRecordsAfter) / numRecordsBefore);

%% Extract filtered diameters
filteredDiameters = [filteredData.bubble_diameter];

%% Create histograms for filtered data
% Full histogram
[filteredFullCounts, ~] = histcounts(filteredDiameters, fullHistEdges);

% Target range histogram
filteredInRange = filteredDiameters(filteredDiameters >= targetRange(1) & filteredDiameters <= targetRange(2));
[filteredCounts, ~] = histcounts(filteredInRange, histEdges);

%% Plot filtered histogram for full dataset
figure('Name', 'Full Bubble Distribution After Filtering', 'Position', [100, 100, 1000, 500]);
histogram('BinEdges', fullHistEdges, 'BinCounts', filteredFullCounts);
title('Full Bubble Diameter Distribution (After Filtering)');
xlabel('Bubble Diameter (m)');
ylabel('Count');
grid on;

% Use the same axis limits as before filtering
xlim(fullXLim);
ylim(fullYLim);

% Highlight the target range
hold on;
xline(targetRange(1), 'g-', 'LineWidth', 1.5);
xline(targetRange(2), 'g-', 'LineWidth', 1.5);
text(targetRange(1), 0.9*max(ylim), 'Target Range Start', 'Color', 'g', 'HorizontalAlignment', 'left');
text(targetRange(2), 0.9*max(ylim), 'Target Range End', 'Color', 'g', 'HorizontalAlignment', 'right');
hold off;

%% Create a before/after comparison figure for the full dataset
figure('Name', 'Full Dataset Before vs After Comparison', 'Position', [100, 100, 1000, 800]);

% Before filtering
subplot(2, 1, 1);
histogram('BinEdges', fullHistEdges, 'BinCounts', fullHistCounts);
title('Before Filtering (Full Dataset)');
xlabel('Bubble Diameter (m)');
ylabel('Count');
grid on;
xlim(fullXLim);
ylim(fullYLim);

% Highlight the target range
hold on;
xline(targetRange(1), 'r-', 'LineWidth', 1.5);
xline(targetRange(2), 'r-', 'LineWidth', 1.5);
text(targetRange(1), 0.9*max(ylim), 'Target Range', 'Color', 'r', 'HorizontalAlignment', 'left');
hold off;

% After filtering
subplot(2, 1, 2);
histogram('BinEdges', fullHistEdges, 'BinCounts', filteredFullCounts);
title('After Filtering (Full Dataset)');
xlabel('Bubble Diameter (m)');
ylabel('Count');
grid on;
xlim(fullXLim);
ylim(fullYLim);

% Highlight the target range
hold on;
xline(targetRange(1), 'g-', 'LineWidth', 1.5);
xline(targetRange(2), 'g-', 'LineWidth', 1.5);
text(targetRange(1), 0.9*max(ylim), 'Target Range', 'Color', 'g', 'HorizontalAlignment', 'left');
hold off;

%% Create a before/after comparison figure for the target range
figure('Name', 'Target Range Before vs After Comparison', 'Position', [100, 100, 1000, 800]);

% Determine axis limits for consistent display
targetRangeYMax = max(max(histCounts), max(filteredCounts)) * 1.1;

% Before filtering
subplot(2, 1, 1);
histogram('BinEdges', histEdges, 'BinCounts', histCounts);
title('Before Filtering (Target Range)');
xlabel('Bubble Diameter (m)');
ylabel('Count');
grid on;
xlim([targetRange(1), targetRange(2)]);
ylim([0, targetRangeYMax]);

% Add threshold line
hold on;
plot([histEdges(1), histEdges(end)], [targetBinCount, targetBinCount], 'r--', 'LineWidth', 1.5);
text(histEdges(1), targetBinCount*1.05, sprintf('Target: %d', targetBinCount), ...
    'Color', 'r', 'FontWeight', 'bold');

% Highlight bins that were filtered
for i = 1:numel(binsToFilter)
    binIdx = binsToFilter(i);
    x = [histEdges(binIdx), histEdges(binIdx+1), histEdges(binIdx+1), histEdges(binIdx)];
    y = [0, 0, histCounts(binIdx), histCounts(binIdx)];
    patch(x, y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
hold off;

% After filtering
subplot(2, 1, 2);
histogram('BinEdges', histEdges, 'BinCounts', filteredCounts);
title('After Filtering (Target Range)');
xlabel('Bubble Diameter (m)');
ylabel('Count');
grid on;
xlim([targetRange(1), targetRange(2)]);
ylim([0, targetRangeYMax]);

% Add threshold line
hold on;
plot([histEdges(1), histEdges(end)], [targetBinCount, targetBinCount], 'g--', 'LineWidth', 1.5);
text(histEdges(1), targetBinCount*1.05, sprintf('Target: %d', targetBinCount), ...
    'Color', 'g', 'FontWeight', 'bold');
hold off;

%% Update the data variable in S with the filtered data
S.(varName) = filteredData;

%% Save the filtered data as the new aggregated_results.mat
save(origMatFile, '-struct', 'S', '-v7');
fprintf('\nSaved filtered data to %s\n', origMatFile);

%% Create a comparison bar chart
figure('Name', 'Bin-by-Bin Comparison', 'Position', [100, 100, 1200, 600]);

% Plot the comparison as a grouped bar chart
bar([histCounts', filteredCounts']);
title('Before vs After Filtering (Target Range Only)');
xlabel('Bin Number');
ylabel('Count');
grid on;

% Add a reference line for the target count
hold on;
plot([0, numel(histCounts)+1], [targetBinCount, targetBinCount], 'r--', 'LineWidth', 1.5);
hold off;

% Add legend
legend('Before Filtering', 'After Filtering', 'Target Threshold');

% Adjust the display to show every 5th bin label
binLabels = cell(numel(binCenters), 1);
for i = 1:numel(binCenters)
    if mod(i, 5) == 0
        binLabels{i} = sprintf('%.4f', binCenters(i));
    else
        binLabels{i} = '';
    end
end

% Set the bin labels
xticks(1:numel(binCenters));
xticklabels(binLabels);
xtickangle(90);

% Highlight bins that were filtered
for i = 1:numel(binsToFilter)
    binIdx = binsToFilter(i);
    highlight_x = binIdx;
    highlight_y = max(histCounts(binIdx), filteredCounts(binIdx));
    
    % Add a highlight marker
    hold on;
    plot(highlight_x, highlight_y * 1.05, 'v', 'MarkerSize', 8, ...
        'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'none');
    hold off;
end