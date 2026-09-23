%% pick_measurement_points.m
% README — INTERACTIVELY PICK PIXELS FOR MEASUREMENT
% ==================================================
%
% WHAT THIS DOES
%   1. Shows your photo (distorted image, same as calibration).
%   2. You click reference corners, then object corners (ginput).
%   3. Saves picked_measurement_points.mat and measurement_points.json
%      for measure_object_calibrated.m / measure_object_calibrated.py.
%
% BEFORE YOU RUN
%   - Run Calibration1.m once (cameraParams.mat exists).
%   - Set imagePath and referenceWorldXY below (world units = inches for HW1).
%   - Click corners in the SAME ORDER as the rows in referenceWorldXY.
%
% HOW TO RUN (MATLAB — needs a display window)
%   cd("/Users/home/GSU/Computer_Vision/Computer_Vision/HW1")
%   pick_measurement_points
%
% THEN MEASURE
%   measure_object_calibrated          % MATLAB (auto-loads picked points)
%   python3 measure_object_calibrated.py   % Python (reads measurement_points.json)
%
% TIPS
%   - Use a photo where the reference and object lie on the same flat surface.
%   - Press Enter after the last click in each step.
%   - Right-click or Enter without clicking can stop ginput early — click exactly N times.

scriptDir = fileparts(mfilename("fullpath"));
if strlength(scriptDir) == 0
    scriptDir = pwd;
end

%% ======================== USER CONFIG =====================================

% Default: first image from your calibration set (change if needed).
imagePath = "/Users/home/Downloads/IMG_5270.jpg";

% Known world (X,Y) on the plane for each reference corner you will click (inches).
referenceWorldXY = [
    0,  0;
    10, 0;
    10, 7;
    0,  7
];

numReferenceClicks = size(referenceWorldXY, 1);
numObjectClicks = 4;

closedPolygon = true;

%% ======================== Load world units from calibration ================

calMatPath = fullfile(scriptDir, "cameraParams.mat");
if isfile(calMatPath)
    loaded = load(calMatPath, "cameraParams");
    worldUnits = string(loaded.cameraParams.WorldUnits);
else
    worldUnits = "inches";
    warning("pick_measurement_points:NoCal", ...
        "cameraParams.mat not found — using worldUnits = inches.");
end

if ~isfile(imagePath)
    error("pick_measurement_points:NoImage", "Image not found: %s", imagePath);
end

if numReferenceClicks ~= size(referenceWorldXY, 1)
    error("pick_measurement_points:Count", ...
        "numReferenceClicks must match rows in referenceWorldXY.");
end

%% ======================== Pick points =====================================

img = imread(imagePath);
fig = figure("Name", "Pick measurement points", "NumberTitle", "off");
imshow(img);
axis("image");
title(sprintf("Step 1/%d: Click %d REFERENCE corners (green). Press Enter when done.", ...
    2, numReferenceClicks), "FontSize", 12);

[xRef, yRef] = ginput(numReferenceClicks);
referenceImageUV_distorted = [xRef, yRef];
hold on;
plot(xRef, yRef, "go-", "LineWidth", 2, "MarkerSize", 10, "MarkerFaceColor", "g");

title(sprintf("Step 2/%d: Click %d OBJECT corners (red). Press Enter when done.", ...
    2, numObjectClicks), "FontSize", 12);
[xObj, yObj] = ginput(numObjectClicks);
objectImageUV_distorted = [xObj, yObj];
plot(xObj, yObj, "r.-", "LineWidth", 2, "MarkerSize", 12);

legend("Reference", "Object", "Location", "best");
drawnow;

fprintf("\n--- Picked distorted pixels (u, v) ---\n");
fprintf("Reference:\n");
disp(referenceImageUV_distorted);
fprintf("Object:\n");
disp(objectImageUV_distorted);

%% ======================== Save for MATLAB + Python ========================

matOut = fullfile(scriptDir, "picked_measurement_points.mat");
save(matOut, ...
    "imagePath", ...
    "referenceWorldXY", ...
    "referenceImageUV_distorted", ...
    "objectImageUV_distorted", ...
    "closedPolygon", ...
    "worldUnits");
fprintf("Saved: %s\n", matOut);

jsonOut = fullfile(scriptDir, "measurement_points.json");
payload = struct();
payload.image_path = char(imagePath);
payload.world_units = char(worldUnits);
payload.closed_polygon = closedPolygon;
payload.reference_world_xy = referenceWorldXY;
payload.reference_image_uv = referenceImageUV_distorted;
payload.object_image_uv = objectImageUV_distorted;
payload.calibration_file = "camera_calibration.json";
payload.notes = "Written by pick_measurement_points.m — distorted pixel coordinates.";

fid = fopen(jsonOut, "w");
if fid < 0
    error("pick_measurement_points:JsonWrite", "Could not write %s", jsonOut);
end
fprintf(fid, "%s\n", jsonencode(payload));
fclose(fid);
fprintf("Saved: %s\n", jsonOut);

fprintf("\nNext: run measure_object_calibrated (MATLAB) or measure_object_calibrated.py\n");
