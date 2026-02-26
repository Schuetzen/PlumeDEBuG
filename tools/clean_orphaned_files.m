%---------------------------------------------------------------------
% Script: cleanupExtraFiles.m
%
% Description:
%   This script cleans up extra files in the RealBubbleDB directories that
%   are not referenced in the aggregated results file.
%
%   It performs the following steps:
%     1. Loads aggregated_results.mat and extracts file prefixes from the
%        aggregated data (expects variable 'imgInfo' or 'combinedImgInfo').
%     2. Ensures that each extracted prefix is a character vector.
%     3. Extracts file prefixes from the Cropped folder (files ending with
%        '_cropped.tif').
%     4. Extracts file prefixes from the Masks folder (files ending with '_mask.tif').
%     5. Deletes any files in the Cropped and Masks folders that do not have a
%        corresponding prefix in the aggregated results.
%
% Instructions:
%   - Ensure that the aggregated_results.mat file exists and contains the expected
%     variable.
%   - Verify that the Cropped and Masks folders exist and follow the naming conventions.
%   - Run this script in MATLAB.
%
% Dependencies:
%   - MATLAB with support for functions such as dir, fullfile, exist, and delete.
%
% Author:rf
%   Schuetzen Jung (modified)
%
% Date:
%   02-26-2025
%---------------------------------------------------------------------

%% 1. Set the base folder (modify this variable to change the root folder)
baseFolder = '020';

%% 2. Define file paths using baseFolder
matFile    = fullfile(baseFolder, 'aggregated_results.mat');
croppedDir = fullfile(baseFolder, 'Dataset', 'Cropped');
masksDir   = fullfile(baseFolder, 'Dataset', 'Masks');

%% 3. Load aggregated_results.mat and extract prefixes
if ~exist(matFile, 'file')
    error('Cannot find file %s', matFile);
end

S = load(matFile);

% Determine which variable to use
if isfield(S, 'combinedImgInfo')
    data = S.combinedImgInfo;
elseif isfield(S, 'imgInfo')
    data = S.imgInfo;
else
    error('aggregated_results.mat does not contain variable imgInfo or combinedImgInfo.');
end

% Extract image names from data (supports cell array or structure array)
if iscell(data)
    aggregatedPrefixes = cellfun(@(x) x.imageName, data, 'UniformOutput', false);
elseif isstruct(data)
    aggregatedPrefixes = {data.imageName};
else
    error('imgInfo is neither a cell array nor a structure array.');
end

%% 4. Ensure all entries in aggregatedPrefixes are character vectors
for k = 1:numel(aggregatedPrefixes)
    if ~ischar(aggregatedPrefixes{k})
        if isempty(aggregatedPrefixes{k})
            aggregatedPrefixes{k} = '';
        else
            aggregatedPrefixes{k} = num2str(aggregatedPrefixes{k});
        end
    end
end

% Remove duplicate prefixes
aggregatedPrefixes = unique(aggregatedPrefixes);

%% 5. Extract prefixes from the Cropped folder
croppedFiles = dir(fullfile(croppedDir, '*_cropped.tif'));

croppedPrefixes = {};
suffixCropped = '_cropped.tif';
for i = 1:length(croppedFiles)
    fname = croppedFiles(i).name;
    if length(fname) >= length(suffixCropped) && strcmp(fname(end-length(suffixCropped)+1:end), suffixCropped)
        prefix = fname(1:end-length(suffixCropped));
        croppedPrefixes{end+1} = prefix;
    else
        fprintf('Cropped file "%s" does not follow the expected naming convention.\n', fname);
    end
end

%% 6. Extract prefixes from the Masks folder
maskFiles = dir(fullfile(masksDir, '*_mask.tif'));

masksPrefixes = {};
suffixMask = '_mask.tif';
for i = 1:length(maskFiles)
    fname = maskFiles(i).name;
    if length(fname) >= length(suffixMask) && strcmp(fname(end-length(suffixMask)+1:end), suffixMask)
        prefix = fname(1:end-length(suffixMask));
        masksPrefixes{end+1} = prefix;
    else
        fprintf('Masks file "%s" does not follow the expected naming convention.\n', fname);
    end
end

%% 7. Display the extracted prefixes for verification
fprintf('Aggregated prefixes:\n');
disp(aggregatedPrefixes);
fprintf('Cropped prefixes:\n');
disp(croppedPrefixes);
fprintf('Masks prefixes:\n');
disp(masksPrefixes);

%% 8. Delete extra files from the Cropped folder that are not in aggregatedPrefixes
for i = 1:length(croppedPrefixes)
    prefix = croppedPrefixes{i};
    if ~any(strcmp(aggregatedPrefixes, prefix))
        croppedFile = fullfile(croppedDir, [prefix, suffixCropped]);
        if exist(croppedFile, 'file')
            delete(croppedFile);
            fprintf('Deleted extra file in Cropped folder: %s\n', croppedFile);
        end
    end
end

%% 9. Delete extra files from the Masks folder that are not in aggregatedPrefixes
for i = 1:length(masksPrefixes)
    prefix = masksPrefixes{i};
    if ~any(strcmp(aggregatedPrefixes, prefix))
        maskFile = fullfile(masksDir, [prefix, suffixMask]);
        if exist(maskFile, 'file')
            delete(maskFile);
            fprintf('Deleted extra file in Masks folder: %s\n', maskFile);
        end
    end
end

fprintf('\nCleanup complete.\n');
