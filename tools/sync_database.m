%---------------------------------------------------------------------
% Script: syncAndCleanupData_v2.m
%
% Description:
%   This script synchronizes files among the aggregated results, Cropped
%   images, and Mask images. It identifies the common set of files present
%   in all three locations and removes any entries or files that are not
%   part of this common set. This ensures that every cropped image and
%   mask has a corresponding entry in the .mat file, and vice-versa.
%
%   Key Features:
%     - Handles imageName entries in the .mat file with or without '.tif'.
%     - Finds the intersection of prefixes from all three sources.
%     - Deletes orphaned files from Cropped and Masks folders.
%     - Removes orphaned data entries from the .mat file and saves it.
%
% Author:
%   Schuetzen Jung (modified for synchronization and robustness)
%
% Date:
%   08-19-2025
%---------------------------------------------------------------------

%% 1. Set the base folder (modify this variable)
% --- Update folder ---
baseFolder = '../dataset/0442/';

%% 2. Define file paths and suffixes
matFile    = fullfile(baseFolder, 'aggregated_results.mat');
croppedDir = fullfile(baseFolder, 'Dataset', 'Cropped');
masksDir   = fullfile(baseFolder, 'Dataset', 'Masks');
suffixCropped = '_cropped.tif';
suffixMask = '_mask.tif';

%% 3. Load aggregated_results.mat and extract clean prefixes
fprintf('--> 1. Loading data from %s...\n', matFile);
if ~exist(matFile, 'file')
    error('Cannot find file: %s', matFile);
end
S = load(matFile);

% Determine which variable to use
if isfield(S, 'combinedImgInfo')
    data = S.combinedImgInfo;
    dataFieldName = 'combinedImgInfo';
elseif isfield(S, 'imgInfo')
    data = S.imgInfo;
    dataFieldName = 'imgInfo';
else
    error('aggregated_results.mat does not contain "imgInfo" or "combinedImgInfo".');
end

% Extract image names from the data structure
if iscell(data)
    aggregatedPrefixes = cellfun(@(x) x.imageName, data, 'UniformOutput', false);
elseif isstruct(data)
    aggregatedPrefixes = {data.imageName};
else
    error('The data is neither a cell array nor a structure array.');
end

% --- New feature: strip .tif suffix from imageName entries ---
% This ensures consistency before comparison.
fprintf('   Normalizing imageName entries by removing any ".tif" suffix...\n');
aggregatedPrefixes = cellfun(@(x) char(x), aggregatedPrefixes, 'UniformOutput', false);
aggregatedPrefixes = cellfun(@(x) regexprep(x, '\.tif$', ''), aggregatedPrefixes, 'UniformOutput', false);

% Remove duplicates after cleaning
aggregatedPrefixes = unique(aggregatedPrefixes);

%% 4. Extract prefixes from the Cropped folder
fprintf('--> 2. Scanning "Cropped" folder...\n');
croppedFiles = dir(fullfile(croppedDir, ['*', suffixCropped]));
croppedPrefixes = cell(1, length(croppedFiles));
for i = 1:length(croppedFiles)
    croppedPrefixes{i} = erase(croppedFiles(i).name, suffixCropped);
end
croppedPrefixes = unique(croppedPrefixes);

%% 5. Extract prefixes from the Masks folder
fprintf('--> 3. Scanning "Masks" folder...\n');
maskFiles = dir(fullfile(masksDir, ['*', suffixMask]));
masksPrefixes = cell(1, length(maskFiles));
for i = 1:length(maskFiles)
    masksPrefixes{i} = erase(maskFiles(i).name, suffixMask);
end
masksPrefixes = unique(masksPrefixes);

%% 6. Find the common set of prefixes (the intersection)
% This is the core of the synchronization logic. We find prefixes
% that exist in ALL THREE locations.
commonPrefixes = intersect(aggregatedPrefixes, croppedPrefixes);
commonPrefixes = intersect(commonPrefixes, masksPrefixes);

fprintf('\n--- Analysis Summary ---\n');
fprintf('Found %d unique entries in aggregated_results.mat.\n', numel(aggregatedPrefixes));
fprintf('Found %d unique images in Cropped folder.\n', numel(croppedPrefixes));
fprintf('Found %d unique images in Masks folder.\n', numel(masksPrefixes));
fprintf('=> Found %d common files to keep for synchronization.\n\n', numel(commonPrefixes));

%% 7. Clean the Cropped folder
fprintf('--> 4. Cleaning "Cropped" folder...\n');
prefixesToDelete_cropped = setdiff(croppedPrefixes, commonPrefixes);
if isempty(prefixesToDelete_cropped)
    fprintf('   "Cropped" folder is already synchronized. No files to delete.\n');
else
    fprintf('   Found %d extra files to delete:\n', numel(prefixesToDelete_cropped));
    for i = 1:length(prefixesToDelete_cropped)
        prefix = prefixesToDelete_cropped{i};
        fileToDelete = fullfile(croppedDir, [prefix, suffixCropped]);
        if exist(fileToDelete, 'file')
            delete(fileToDelete);
            fprintf('     - Deleted: %s\n', [prefix, suffixCropped]);
        end
    end
end

%% 8. Clean the Masks folder
fprintf('--> 5. Cleaning "Masks" folder...\n');
prefixesToDelete_masks = setdiff(masksPrefixes, commonPrefixes);
if isempty(prefixesToDelete_masks)
    fprintf('   "Masks" folder is already synchronized. No files to delete.\n');
else
    fprintf('   Found %d extra files to delete:\n', numel(prefixesToDelete_masks));
    for i = 1:length(prefixesToDelete_masks)
        prefix = prefixesToDelete_masks{i};
        fileToDelete = fullfile(masksDir, [prefix, suffixMask]);
        if exist(fileToDelete, 'file')
            delete(fileToDelete);
            fprintf('     - Deleted: %s\n', [prefix, suffixMask]);
        end
    end
end

%% 9. Clean and update the aggregated_results.mat file
fprintf('--> 6. Cleaning aggregated_results.mat...\n');
prefixesToRemove_agg = setdiff(aggregatedPrefixes, commonPrefixes);
if isempty(prefixesToRemove_agg)
    fprintf('   .mat file is already synchronized. No update needed.\n');
else
    fprintf('   Found %d extra entries to remove. Updating file...\n', numel(prefixesToRemove_agg));
    
    % Create a logical index of entries to keep. We must also clean the
    % imageName from the original data `data` before comparing.
    if iscell(data)
        imageNamesInData_cleaned = cellfun(@(x) regexprep(char(x.imageName), '\.tif$', ''), data, 'UniformOutput', false);
        keepIdx = ismember(imageNamesInData_cleaned, commonPrefixes);
    else % isstruct
        imageNamesInData = cellfun(@(x) char(x), {data.imageName}, 'UniformOutput', false);
        imageNamesInData_cleaned = cellfun(@(x) regexprep(x, '\.tif$', ''), imageNamesInData, 'UniformOutput', false);
        keepIdx = ismember(imageNamesInData_cleaned, commonPrefixes);
    end

    % Filter the data using the logical index
    filteredData = data(keepIdx);
    
    % Update the variable in the original loaded structure
    S.(dataFieldName) = filteredData;
    
    % Save the updated structure back to the .mat file, overwriting it
    save(matFile, '-struct', 'S');
    fprintf('   Successfully updated and saved %s.\n', matFile);
end

fprintf('\nSynchronization complete. All locations are now aligned. ✅\n');