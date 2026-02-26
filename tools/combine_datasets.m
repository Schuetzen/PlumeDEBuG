function imgInfo = QuickCombine()
    %% Define the base directory and source folders
    baseDir = '../dataset';  % base directory
    sourceDirs = {'044','041','020','101', '401', '406','407', '408', '409','420','421','422', '0442','600','601'};
    
    %% Define the destination folder structure
    destFolder = '../Aggregated_bubble_data/';
    if ~exist(destFolder, 'dir')
        mkdir(destFolder);
    end
    
    %% Initialize the variable for imgInfo data
    imgInfo = [];
    
    %% Loop over each source folder
    for i = 1:length(sourceDirs)
        currDir = sourceDirs{i};
        
        %% Load and combine the aggregated_results.mat file
        matFile = fullfile(baseDir, currDir, 'aggregated_results.mat');  % full path
        
        if exist(matFile, 'file')
            data = load(matFile);
            if isfield(data, 'imgInfo')
                disp(['Size of imgInfo in folder ' currDir ': ' mat2str(size(data.imgInfo))]);
                currImgInfo = data.imgInfo(:);
                
                for k = 1:length(currImgInfo)
                    currImgInfo(k).folderPath = currDir;
                    if isfield(currImgInfo(k), 'imageName') && contains(currImgInfo(k).imageName, '.tif')
                        currImgInfo(k).imageName = strrep(currImgInfo(k).imageName, '.tif', '');
                    end
                end
                
                imgInfo = [imgInfo; currImgInfo];
            else
                warning('File %s does not contain variable ''imgInfo''.', matFile);
            end
        else
            warning('File %s does not exist.', matFile);
        end
    end
    
    %% Save the combined aggregated_results.mat file
    save(fullfile(destFolder, 'aggregated_results.mat'), 'imgInfo');
    fprintf('Processing complete. Total entries: %d\n', length(imgInfo));
end