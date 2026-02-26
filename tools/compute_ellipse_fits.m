function batchEllipseAnalysis()
    %% Define base directory and source folders
    baseDir = '../dataset';
    sourceDirs = {'044','041','020','101', '401', '406','407', '408', '409','420','421','422', '0442','600','601'};
    
    %% Load resolution file
    scriptFull = mfilename('fullpath');
    scriptDir  = fileparts(scriptFull);
    resPath    = fullfile(scriptDir, '../Resolution.mat');
    if ~exist(resPath, 'file')
        error('Resolution file not found at: %s', resPath);
    end
    R = load(resPath);
    if isfield(R, 'res')
        res = R.res;
    elseif isfield(R, 'resolution')
        res = R.resolution;
    else
        error('Resolution variable not found in %s', resPath);
    end
    
    %% Iterate over each folder
    for i = 1:length(sourceDirs)
        currDir = sourceDirs{i};
        
        % Define paths
        maskFolder = fullfile(baseDir, currDir, 'Dataset/Masks');  % adjust to actual mask path if needed
        matFile = fullfile(baseDir, currDir, 'aggregated_results.mat');
        
        fprintf('\n========== Processing folder: %s ==========\n', currDir);
        
        % Check files exist
        if ~exist(matFile, 'file')
            warning('aggregated_results.mat not found in %s, skipping...', currDir);
            continue;
        end
        if ~exist(maskFolder, 'dir')
            warning('Masks folder not found in %s, skipping...', currDir);
            continue;
        end
        
        % Process this folder
        try
            processFolder(maskFolder, matFile, res);
            fprintf('Successfully processed: %s\n', currDir);
        catch ME
            warning('Error processing %s: %s', currDir, ME.message);
        end
    end
    
    fprintf('\n========== Batch processing complete ==========\n');
end

function processFolder(maskFolder, matFile, res)
    % Load aggregated_results
    S = load(matFile);
    vars = fieldnames(S);
    
    % Find the struct containing imageName
    targetVar = '';
    aggregated_results = [];
    for k = 1:numel(vars)
        v = S.(vars{k});
        if isstruct(v) && isfield(v, 'imageName')
            aggregated_results = v;
            targetVar = vars{k};
            break;
        end
    end
    
    if isempty(aggregated_results)
        error('No struct array with field ''imageName'' found');
    end
    
    n = numel(aggregated_results);
    fprintf('Updating %d entries...\n', n);
    
    for i = 1:n
        name = aggregated_results(i).imageName;
        [~, baseName, ~] = fileparts(name);
        
        maskFile = fullfile(maskFolder, [baseName '_mask.tif']);
        if ~exist(maskFile, 'file')
            % Try alternative naming format
            maskFile = fullfile(maskFolder, [baseName '.tif']);
        end
        
        if ~exist(maskFile, 'file')
            warning('Mask not found for %s', name);
            continue;
        end
        
        try
            img   = imread(maskFile);
            stats = getImageStats(img);
            
            % Compute ellipse parameters
            major_px   = stats.MajorAxisLength;
            minor_px   = stats.MinorAxisLength;
            axis_ratio = minor_px / major_px;
            ellipse_d_px = (8 * (minor_px/2) * (major_px/2)^2)^(1/3);
            ellipse_d_m  = ellipse_d_px * res;
            
            % Update/replace fields
            aggregated_results(i).Area          = stats.Area;
            aggregated_results(i).MajorAxis_px  = major_px;
            aggregated_results(i).MinorAxis_px  = minor_px;
            aggregated_results(i).AxisRatio     = axis_ratio;
            aggregated_results(i).ellipse_d_px  = ellipse_d_px;
            aggregated_results(i).ellipse_d_m   = ellipse_d_m;
            
        catch ME
            fprintf('Error processing %s: %s\n', name, ME.message);
        end
    end
    
    % Save back to original file
    S.(targetVar) = aggregated_results;
    save(matFile, '-struct', 'S');
    fprintf('Saved to: %s\n', matFile);
end

function stats = getImageStats(img)
    if ~islogical(img)
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        img = imbinarize(img);
    end
    [y, x] = find(img);
    if isempty(x)
        stats = struct('Area',0, 'MajorAxisLength',0, 'MinorAxisLength',0);
        return;
    end
    area  = numel(x);
    x_c   = x - mean(x);
    y_c   = y - mean(y);
    M20   = mean(x_c.^2);
    M02   = mean(y_c.^2);
    M11   = mean(x_c .* y_c);
    C     = [M20, M11; M11, M02];
    [~, D] = eig(C);
    ev    = diag(D);
    ev    = sort(ev, 'descend');
    major = 4 * sqrt(ev(1));
    minor = 4 * sqrt(ev(2));
    stats = struct('Area', area, 'MajorAxisLength', major, 'MinorAxisLength', minor);
end