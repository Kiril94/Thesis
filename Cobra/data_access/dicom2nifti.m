scan_path = 'Z:\\positive\\f430460b276e6618ad2c73daf269c228';
listing = dir(scan_path);
subdir = listing([listing.isdir]);
disp(a);
for k = 1 : length(subdir)
  fprintf('Sub folder #%d = %s\n', k, subdir(k).name);
end
example_folder = fullfile(scan_path, subdir(3).name);
subsubfolder = dir(example_folder);
subsubfolder_name = dir(example_folder);
disp();
%disp(example_folder);
%dicom_file_paths = {};