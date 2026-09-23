%% calibrate_from_folder.m
% README — STEP 1: CAMERA CALIBRATION (MATLAB, 20+ IMAGES IN FOLDER)
% ===================================================================
%
% WHAT THIS DOES
%   Loads all JPEG/PNG images from calibration_images/, detects checkerboard
%   corners, estimates cameraParameters, saves cameraParams.mat and
%   camera_calibration.json (same outputs as Calibration1.m).
%
% SETUP
%   Put at least 20 smartphone checkerboard photos in:
%     HW1/calibration_images/
%   Edit calibration_config.json (via square size in this script) if needed.
%
% HOW TO RUN (MATLAB)
%   cd("/Users/home/GSU/Computer_Vision/Computer_Vision/HW1")
%   calibrate_from_folder
%
% OUTPUT
%   cameraParams.mat, camera_calibration.json, calibration_report.mat

scriptDir = fileparts(mfilename("fullpath"));
if strlength(scriptDir) == 0
    scriptDir = pwd;
end

imageDir = fullfile(scriptDir, "calibration_images");
minImages = 20;
squareSize = 1.0;  % inches — match physical checkerboard

if ~isfolder(imageDir)
    error("calibrate_from_folder:NoDir", "Create folder and add images: %s", imageDir);
end

exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"];
imageFileNames = strings(0, 1);
for ext = exts
    listing = dir(fullfile(imageDir, ext));
    for k = 1:numel(listing)
        imageFileNames(end + 1) = fullfile(listing(k).folder, listing(k).name); %#ok<AGROW>
    end
end
imageFileNames = sort(imageFileNames);

if numel(imageFileNames) < minImages
    error("calibrate_from_folder:Need20", ...
        "Need at least %d images in %s (found %d).", minImages, imageDir, numel(imageFileNames));
end

fprintf("Calibrating from %d images in %s\n", numel(imageFileNames), imageDir);

minCornerMetric = 0.15;
[imagePoints, patternDims, imagesUsed] = detectCheckerboardPoints(cellstr(imageFileNames), ...
    MinCornerMetric=minCornerMetric, HighDistortion=false);
imageFileNames = cellstr(imageFileNames(imagesUsed));

if numel(imageFileNames) < minImages
    error("calibrate_from_folder:Detect", ...
        "Checkerboard detected in only %d images (need %d).", numel(imageFileNames), minImages);
end

originalImage = imread(imageFileNames{1});
[mrows, ncols] = size(originalImage, 1:2);
worldPoints = patternWorldPoints("checkerboard", patternDims, squareSize);

[cameraParams, imagesUsed2, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    EstimateSkew=false, EstimateTangentialDistortion=false, ...
    NumRadialDistortionCoefficients=2, WorldUnits="inches", ...
    ImageSize=[mrows ncols]);

matPath = fullfile(scriptDir, "cameraParams.mat");
jsonPath = fullfile(scriptDir, "camera_calibration.json");
reportPath = fullfile(scriptDir, "calibration_report.mat");

save(matPath, "cameraParams", "squareSize", "patternDims", "worldPoints", "imageFileNames");
export_camera_calibration(cameraParams, jsonPath);

report = struct();
report.numImagesUsed = numel(imageFileNames);
report.patternDims = patternDims;
report.meanReprojectionError = cameraParams.MeanReprojectionError;
report.imageFileNames = imageFileNames;
save(reportPath, "report", "estimationErrors");

fprintf("Saved %s\nSaved %s\nMean reprojection error: %.4f px\n", ...
    matPath, jsonPath, cameraParams.MeanReprojectionError);

showReprojectionErrors(cameraParams);
