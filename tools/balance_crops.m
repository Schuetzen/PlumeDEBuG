%---------------------------------------------------------------------
% Script: CropBalance.m
%
% Description:
%   This script processes an aggregated results MAT file from the RealBubbleDB.
%   It loads the MAT file containing the variable 'imgInfo', verifies the
%   existence of corresponding cropped and mask image files for each record, and
%   filters out any records missing one or both files. In addition, it synchronizes
%   the masks folder by removing any mask images that do not have a matching cropped
%   image. Finally, it checks for any extra files in the cropped and masks directories
%   that are not referenced in the aggregated results.
%
% Instructions:
%   1. Set the variable 'baseFolder' to the desired folder (e.g., '045' or another).
%   2. Ensure that the MAT file path and the image directories are correctly set relative
%      to baseFolder.
%   3. Run this script in MATLAB.
%
% Dependencies:
%   - MATLAB with support for fullfile, isfield, and file I/O operations.
%   - A proper RealBubbleDB directory structure with the expected files.
%
% Author:
%   Schuetzen Jung
%
% Date:
%   02-26-2025
%---------------------------------------------------------------------
clear all;

%% Set the base folder (change this value to point to a different folder)
baseFolder = '../dataset/601';

%% Define file paths using baseFolder
matFile = fullfile(baseFolder, 'aggregated_results.mat');
if ~exist(matFile, 'file')
    error('Could not find file %s', matFile);
end

croppedDir = fullfile(baseFolder, 'Dataset', 'Cropped');
masksDir   = fullfile(baseFolder, 'Dataset', 'Masks');

%% Load aggregated_results.mat and extract image titles
S = load(matFile);

if isfield(S, 'imgInfo')
    data = S.imgInfo;
    if iscell(data)
        imageTitles = cellfun(@(x) x.imageName, data, 'UniformOutput', false);
    elseif isstruct(data)
        imageTitles = {data.imageName};
    else
        error('Variable imgInfo is neither a cell array nor a struct array.');
    end
else
    error('aggregated_results.mat does not contain variable imgInfo.');
end

fprintf('=== Checking aggregated_results.mat records ===\n');

validIdx = [];

for i = 1:length(imageTitles)
    prefix = imageTitles{i};
    
    % Remove .tif extension if present
    if endsWith(prefix, '.tif')
        prefix = prefix(1:end-4);
    end
    
    croppedFile = fullfile(croppedDir, [prefix, '_cropped.tif']);
    maskFile = fullfile(masksDir, [prefix, '_mask.tif']);
    
    hasCropped = exist(croppedFile, 'file');
    hasMask = exist(maskFile, 'file');
    
    if ~hasCropped
        fprintf('Missing cropped file for prefix "%s": %s\n', prefix, croppedFile);
    end
    if ~hasMask
        fprintf('Missing mask file for prefix "%s": %s\n', prefix, maskFile);
    end
    
    if hasCropped && hasMask
        validIdx(end+1) = i;
    else
        fprintf('Removing aggregated_results record for prefix "%s" due to missing file(s).\n', prefix);
    end
end

% Update the aggregated results by keeping only valid records
if iscell(S.imgInfo)
    S.imgInfo = S.imgInfo(validIdx);
elseif isstruct(S.imgInfo)
    S.imgInfo = S.imgInfo(validIdx);
end

fprintf('Updated aggregated_results.mat: kept %d of %d records.\n', length(validIdx), length(imageTitles));

save(matFile, '-struct', 'S', '-v7.3');

fprintf('\n=== Synchronizing Masks folder ===\n');

croppedFiles = dir(fullfile(croppedDir, '*_cropped.tif'));
if isempty(croppedFiles)
    fprintf('No *_cropped.tif files found in %s.\n', croppedDir);
else
    croppedPrefixes = {};
    suffixCropped = '_cropped.tif';
    for i = 1:length(croppedFiles)
        fileName = croppedFiles(i).name;
        if length(fileName) > length(suffixCropped) && strcmp(fileName(end-length(suffixCropped)+1:end), suffixCropped)
            croppedPrefixes{end+1} = fileName(1:end-length(suffixCropped));
        else
            fprintf('Cropped file "%s" does not follow the expected naming convention, skipping.\n', fileName);
        end
    end
    croppedPrefixes = unique(croppedPrefixes);
    
    maskFiles = dir(fullfile(masksDir, '*_mask.tif'));
    if isempty(maskFiles)
        fprintf('No *_mask.tif files found in %s.\n', masksDir);
    else
        suffixMask = '_mask.tif';
        for i = 1:length(maskFiles)
            maskFileName = maskFiles(i).name;
            fullMaskPath = fullfile(masksDir, maskFileName);
            
            if length(maskFileName) > length(suffixMask) && strcmp(maskFileName(end-length(suffixMask)+1:end), suffixMask)
                maskPrefix = maskFileName(1:end-length(suffixMask));
            else
                fprintf('File "%s" does not follow the expected naming convention, deleting it.\n', maskFileName);
                delete(fullMaskPath);
                continue;
            end
            
            if ~any(strcmp(croppedPrefixes, maskPrefix))
                delete(fullMaskPath);
                fprintf('Deleted mask image: %s (no corresponding cropped file found)\n', fullMaskPath);
            else
                fprintf('Kept mask image: %s\n', maskFileName);
            end
        end
    end
end

if isfield(S, 'imgInfo')
    if iscell(S.imgInfo)
        updatedPrefixes = cellfun(@(x) x.imageName, S.imgInfo, 'UniformOutput', false);
    elseif isstruct(S.imgInfo)
        updatedPrefixes = {S.imgInfo.imageName};
    end
else
    updatedPrefixes = {};
end

% Remove .tif extension from updated prefixes if present
for i = 1:length(updatedPrefixes)
    if endsWith(updatedPrefixes{i}, '.tif')
        updatedPrefixes{i} = updatedPrefixes{i}(1:end-4);
    end
end

updatedPrefixes = unique(updatedPrefixes);

fprintf('\n=== Checking for extra files not referenced in aggregated_results ===\n');

for i = 1:length(croppedFiles)
    fileName = croppedFiles(i).name;
    if length(fileName) > length(suffixCropped) && strcmp(fileName(end-length(suffixCropped)+1:end), suffixCropped)
        prefix = fileName(1:end-length(suffixCropped));
        if ~any(strcmp(updatedPrefixes, prefix))
            fprintf('Extra cropped file not referenced in aggregated_results: %s\n', fileName);
        end
    end
end

maskFiles = dir(fullfile(masksDir, '*_mask.tif'));
for i = 1:length(maskFiles)
    fileName = maskFiles(i).name;
    if length(fileName) > length(suffixMask) && strcmp(fileName(end-length(suffixMask)+1:end), suffixMask)
        prefix = fileName(1:end-length(suffixMask));
        if ~any(strcmp(updatedPrefixes, prefix))
            fprintf('Extra mask file not referenced in aggregated_results: %s\n', fileName);
        end
    end
end

fprintf('\nSynchronization complete.\n');