function export_hidden_steam_csv(filename, t, T1, T2, H)
% EXPORT_HIDDEN_STEAM_CSV  Save steam-header system data to CSV
%   export_hidden_steam_csv('run1.csv', t, T1, T2, H)

    if nargin < 1 || isempty(filename)
        filename = 'hidden_steam_data.csv';
    end

    % ensure column vectors
    T = table( T1(:), T2(:), H(:), ...
        'VariableNames', {'T1','T2','H'} );

    % write to CSV
    writetable(T, filename);

    fprintf('Saved %d rows to "%s".\n', height(T), filename);
end