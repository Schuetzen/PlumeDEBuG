function BubbleDiameterHistogram()
    %% Load Data
    % Load the .mat file (replace this with the actual file path)
    data = load('../Aggregated_bubble_data/aggregated_results.mat');
    
    % Extract the bubble_diameter values from the structure (adjust field names if necessary)
    bubble_diameters = [data.imgInfo.bubble_diameter];
    
    % Convert bubble diameters to mm (multiply by 1000)
    bubble_diameters_mm = bubble_diameters * 1000;
    
    %% Create figure with Nature-style specifications - slimmer proportions
    figure('Units', 'centimeters', 'Position', [5, 5, 26, 8], 'Color', 'white');
    
    % Create main axes with more space for text
    axes('Position', [0.16, 0.18, 0.68, 0.72]);
    hold on;
    
    %% Create the histogram and normalize it to show probability density
    % Increase the number of bins for better resolution
    numBins = 50; % You can adjust this value for more bins
    
    histogram(bubble_diameters_mm, numBins, 'Normalization', 'pdf');
    
    %% Apply Nature-style formatting
    % Set axis properties
    ax = gca;
    ax.LineWidth = 0.75;       % Thinner axis lines for Nature style
    ax.TickLength = [0.015 0.015]; % Shorter ticks for Nature style
    ax.XAxis.TickLength = [0, 0]; % Remove the ticks on the top axis
    ax.YAxis.TickLength = [0, 0]; % Remove the ticks on the right axis
    
    % Helvetica is preferred by Nature
    set(ax, 'FontName', 'Arial', 'FontSize', 14);
    ax.XTick = get(ax, 'XTick'); % Get current X ticks
    ax.YTick = get(ax, 'YTick'); % Get current Y ticks
    set(gca, 'XAxisLocation', 'bottom', 'YAxisLocation', 'left'); % Position ticks at the bottom and left
    
    % Remove top and right spines
    ax.XColor = 'k'; % Black color for bottom axis
    ax.YColor = 'k'; % Black color for left axis
    % Remove grid lines for cleaner Nature look
    ax.YGrid = 'off';
    ax.XGrid = 'off';
    xlim([0.01,15]);
    xticks(0:5:15);  % Set x-axis ticks at 0.1, 5, 10, 15

    % Add axis labels in Nature style
    xlabel('Bubble Diameter (mm)', 'FontName', 'Arial', 'FontSize', 14);
    ylabel('Probability Density', 'FontName', 'Arial', 'FontSize', 14);
    
    %% Apply Border to the Plot (Make sure the figure is enclosed with a border)
    box on;  % This adds a border around the plot
    
    %% Export figure with Nature specifications
    % Higher resolution for publication quality
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'Renderer', 'painters'); % Vector-based renderer

    % Export as TIFF with high resolution (for journal submission)
    print('-dtiff', '-r600', 'BubbleDiameterDistribution_Nature.tif');
    
    % Save as PDF (often preferred by Nature)
    %print('-dpdf', '-painters', '-r600', 'BubbleDiameterDistribution_Nature.pdf');
    
    hold off;
end
