%% CheckFileCorrespondence.m
% This script checks if the images in aggregated_results.mat, Cropped, and Masks folders
% correspond one-to-one, i.e., every recorded prefix exists in both Cropped and Masks,
% and there are no extra files in Cropped and Masks that are not recorded in the aggregated data.
%
% Expected file formats:
% - Prefixes recorded in aggregated_results.mat are stored in the imageName field of imgInfo
% - Files in Cropped folder: <prefix>_cropped.tif
% - Files in Masks folder: <prefix>_mask.tif

clear all;

%% Set base folder path
baseFolder = '040'; % Can be modified to other paths, e.g., '046' or 'path/to/folder'

%% 1. Load aggregated_results.mat and extract prefixes from aggregated data
matFile = fullfile(baseFolder, 'aggregated_results.mat');
if ~exist(matFile, 'file')
    error('Cannot find file %s', matFile);
end

S = load(matFile);

% Try to extract data from combinedImgInfo or imgInfo
if isfield(S, 'combinedImgInfo')
    data = S.combinedImgInfo;
elseif isfield(S, 'imgInfo')
    data = S.imgInfo;
else
    error('Neither imgInfo nor combinedImgInfo variable found in aggregated_results.mat.');
end

% Extract prefixes from aggregated data (assuming each record has imageName field)
if iscell(data)
    % Assume each cell contains a struct with imageName field
    aggregatedPrefixes = cellfun(@(x) x.imageName, data, 'UniformOutput', false);
elseif isstruct(data)
    aggregatedPrefixes = {data.imageName};
else
    error('imgInfo is neither a cell array nor a struct array.');
end

% Ensure all elements in aggregatedPrefixes are converted to character vectors
for k = 1:numel(aggregatedPrefixes)
    if ~ischar(aggregatedPrefixes{k})
        if isempty(aggregatedPrefixes{k})
            aggregatedPrefixes{k} = ''; % Or use 'NA' as placeholder
        else
            aggregatedPrefixes{k} = num2str(aggregatedPrefixes{k});
        end
    end
end

aggregatedPrefixes = unique(aggregatedPrefixes);

%% 2. Extract prefixes from Cropped folder
croppedDir = fullfile(baseFolder, 'Dataset', 'Cropped');
croppedFiles = dir(fullfile(croppedDir, '*_cropped.tif'));
croppedPrefixes = {};
suffixCropped = '_cropped.tif';

for i = 1:length(croppedFiles)
    fname = croppedFiles(i).name;
    if length(fname) >= length(suffixCropped) && strcmp(fname(end-length(suffixCropped)+1:end), suffixCropped)
        prefix = fname(1:end-length(suffixCropped));
        croppedPrefixes{end+1} = prefix;
    else
        fprintf('Cropped file "%s" does not match expected naming format.\n', fname);
    end
end

croppedPrefixes = unique(croppedPrefixes);

%% 3. Extract prefixes from Masks folder
masksDir = fullfile(baseFolder, 'Dataset', 'Masks');
maskFiles = dir(fullfile(masksDir, '*_mask.tif'));
masksPrefixes = {};
suffixMask = '_mask.tif';

for i = 1:length(maskFiles)
    fname = maskFiles(i).name;
    if length(fname) >= length(suffixMask) && strcmp(fname(end-length(suffixMask)+1:end), suffixMask)
        prefix = fname(1:end-length(suffixMask));
        masksPrefixes{end+1} = prefix;
    else
        fprintf('Masks file "%s" does not match expected naming format.\n', fname);
    end
end

masksPrefixes = unique(masksPrefixes);

%% 4. Check correspondence between aggregated_results and Cropped
fprintf('\n*** Checking aggregated_results vs Cropped ***\n');

% 4.1 Check if prefixes recorded in aggregated data exist in Cropped
for i = 1:length(aggregatedPrefixes)
    prefix = aggregatedPrefixes{i};
    if ~any(strcmp(croppedPrefixes, prefix))
        fprintf('Prefix "%s" from aggregated data not found in Cropped folder: %s_cropped.tif\n', prefix, prefix);
    end
end

% 4.2 Check if Cropped has extra prefixes (not in aggregated data)
for i = 1:length(croppedPrefixes)
    prefix = croppedPrefixes{i};
    if ~any(strcmp(aggregatedPrefixes, prefix))
        fprintf('Extra file in Cropped folder: %s_cropped.tif (prefix "%s" not in aggregated data)\n', prefix, prefix);
    end
end

%% 5. Check correspondence between aggregated_results and Masks
fprintf('\n*** Checking aggregated_results vs Masks ***\n');

% 5.1 Check if prefixes recorded in aggregated data exist in Masks
for i = 1:length(aggregatedPrefixes)
    prefix = aggregatedPrefixes{i};
    if ~any(strcmp(masksPrefixes, prefix))
        fprintf('Prefix "%s" from aggregated data not found in Masks folder: %s_mask.tif\n', prefix, prefix);
    end
end

% 5.2 Check if Masks has extra prefixes
for i = 1:length(masksPrefixes)
    prefix = masksPrefixes{i};
    if ~any(strcmp(aggregatedPrefixes, prefix))
        fprintf('Extra file in Masks folder: %s_mask.tif (prefix "%s" not in aggregated data)\n', prefix, prefix);
    end
end

fprintf('\nCheck completed.\n');