%---------------------------------------------------------------------
% Script: filterimgInfo.m
%
% Description:
%   This script loads a MAT file containing the variable 'imgInfo',
%   filters out any rows (structure elements) for which all fields are empty,
%   and then saves the filtered structure array into a new MAT file.
%
% Instructions:
%   1. Replace 'your_mat_file.mat' with the name of your MAT file if needed.
%   2. Run this script in MATLAB.
%--------------------------------------------------------------------

% Clear workspace and command window (optional)
clear; clc;

% Define the base path to your folder (replace '406' with your desired folder name or path)
basePath = '420';

% Construct the file path for the MAT file containing imgInfo
matFile = fullfile(basePath, 'aggregated_results.mat');

% Load the MAT file containing imgInfo.
load(matFile);  

% Check if imgInfo exists
if ~exist('imgInfo', 'var')
    error('The variable "imgInfo" was not found in the loaded file.');
end

% Determine which rows are "empty" (i.e., all fields are empty)
emptyRows = arrayfun(@(s) all(structfun(@isempty, s)), imgInfo);

% Filter out the empty rows
imgInfo = imgInfo(~emptyRows);

% Construct the output file path for the filtered data
outputFile = fullfile(basePath, 'filtered_mat_file.mat');

% Save the filtered structure to a new MAT file.
save(outputFile, 'imgInfo');

% Display a message to confirm completion.
disp('Filtering complete. The filtered data has been saved to "filtered_mat_file.mat".');
