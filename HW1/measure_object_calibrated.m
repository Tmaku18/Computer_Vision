%% measure_object_calibrated.m
% README — MEASURE REAL 2D OBJECT SIZE USING YOUR CALIBRATION
% ============================================================
%
% WHAT THIS DOES
%   1. Loads cameraParams from Calibration1.m (cameraParams.mat).
%   2. Undistorts pixel clicks using lens parameters (perspective + distortion model).
%   3. Maps undistorted image points to the world plane via homography:
%        lambda * [u; v; 1] = H * [X; Y; 1]
%   4. Reports edge lengths in your calibration units (inches for HW1).
%
% BEFORE YOU RUN
%   1. Run Calibration1.m once (creates cameraParams.mat and camera_calibration.json).
%   2. Pick pixels interactively (recommended):
%        pick_measurement_points
%      This writes picked_measurement_points.mat (auto-loaded below).
%   3. Or fill in USER CONFIG manually (reference + object distorted pixels).
%
% HOW TO RUN (MATLAB)
%   cd("/Users/home/GSU/Computer_Vision/Computer_Vision/HW1")
%   measure_object_calibrated
%
% OPTIONAL
%   measure_planar_object_2d_demo.m — synthetic test without your images.
%
% REQUIRES
%   Computer Vision Toolbox (undistortPoints, fitgeotform2d).

scriptDir = fileparts(mfilename("fullpath"));
if strlength(scriptDir) == 0
    scriptDir = pwd;
end

%% ======================== USER CONFIG (edit this) ========================

% Image you measured (for display only; points are entered as pixels below).
imagePath = fullfile(scriptDir, "..", "data", "measurement_image.jpg");

% At least 4 reference correspondences: known world (X,Y) on the plane ↔ distorted pixels (u,v).
% Example: 10" × 7" reference rectangle on the table (units must match calibration — inches).
referenceWorldXY = [
    0,  0;
    10, 0;
    10, 7;
    0,  7
];

referenceImageUV_distorted = [
    312.4, 428.1;
    891.2, 401.6;
    920.5, 812.3;
    285.0, 845.9
];

% Object corners to measure (distorted pixels), in order around the shape.
objectImageUV_distorted = [
    420.0, 510.0;
    720.0, 495.0;
    735.0, 680.0;
    405.0, 695.0
];

closedPolygon = true;

%% ======================== Auto-load picked points =========================

pickedPath = fullfile(scriptDir, "picked_measurement_points.mat");
if isfile(pickedPath)
    picked = load(pickedPath);
    imagePath = picked.imagePath;
    referenceWorldXY = picked.referenceWorldXY;
    referenceImageUV_distorted = picked.referenceImageUV_distorted;
    objectImageUV_distorted = picked.objectImageUV_distorted;
    if isfield(picked, "closedPolygon")
        closedPolygon = picked.closedPolygon;
    end
    fprintf("Using picked points from %s\n", pickedPath);
end

%% ======================== Load calibration ================================

calMatPath = fullfile(scriptDir, "cameraParams.mat");
if ~isfile(calMatPath)
    error("measure_object_calibrated:NoCalibration", ...
        "Missing %s — run Calibration1.m first.", calMatPath);
end

loaded = load(calMatPath, "cameraParams");
cameraParams = loaded.cameraParams;
worldUnits = string(cameraParams.WorldUnits);

fprintf("Loaded calibration (%s). Image size: [%d %d]\n", ...
    worldUnits, cameraParams.ImageSize(1), cameraParams.ImageSize(2));

%% ======================== Homography + measure ============================
% measure_planar_object_2d undistorts pixels when cameraParams is provided (5th argument).

results = measure_planar_object_2d( ...
    referenceWorldXY, ...
    referenceImageUV_distorted, ...
    objectImageUV_distorted, ...
    worldUnits, ...
    cameraParams);

fprintf("\n--- Measurement results (%s) ---\n", worldUnits);
fprintf("World XY (object corners):\n");
disp(results.worldXY);
fprintf("Edge lengths: %s\n", mat2str(results.edgeLengths, 5));
fprintf("Perimeter: %.4f %s\n", results.perimeter, worldUnits);

if size(objectImageUV_distorted, 1) >= 4 && numel(results.edgeLengths) >= 2
    fprintf("Approx. width x height (edges 1 and 2): %.4f x %.4f %s\n", ...
        results.edgeLengths(1), results.edgeLengths(2), worldUnits);
end

%% ======================== Optional visualization ==========================

if isfile(imagePath)
    img = imread(imagePath);
    figure("Name", "Measurement overlay");
    imshow(img);
    hold on;
    plot(referenceImageUV_distorted(:, 1), referenceImageUV_distorted(:, 2), "go-", "LineWidth", 1.5);
    plot(objectImageUV_distorted(:, 1), objectImageUV_distorted(:, 2), "r.-", "LineWidth", 2);
    legend("Reference (distorted px)", "Object (distorted px)", Location="best");
    title(sprintf("Planar measure (%s) — see Command Window for lengths", worldUnits));
    hold off;
else
    fprintf("Skip display: image not found at %s\n", imagePath);
end
