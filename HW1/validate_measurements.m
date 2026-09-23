%% validate_measurements.m
% README — STEP 3: VALIDATE 20 MEASUREMENTS + ERROR STATISTICS (MATLAB)
% =====================================================================
%
% MATLAB counterpart of validate_measurements.py. Uses the same files.
%
% HOW TO RUN
%   cd("/Users/home/GSU/Computer_Vision/Computer_Vision/HW1")
%   validate_measurements                                   % raw-pixel mode (B)
%   validate_measurements("validation_measurements_export.csv")  % web export (A)
%
% INPUT (either of)
%   A) CSV exported from the web app (Module 2 > Step 3 > Export CSV). It
%      already holds est_homography / est_pinhole; only statistics are computed.
%   B) validation_config.json + validation_measurements.csv with raw pixels:
%        image_path          validation photo (relative to the repo root)
%        camera_distance_m   taped camera-to-plane distance (> 2 m)
%        unit                unit of ground truth and reference ("in", "cm", ...)
%        reference           {"type":"checkerboard","pattern":[9,6],"square_size":1}
%                         or {"type":"rectangle","width":W,"height":H,
%                             "image_uv":[[u,v] x4 TL, TR, BR, BL]}
%      CSV columns: measurement_id, ground_truth, u1, v1, u2, v2, description
%      Requires cameraParams.mat (calibrate_from_folder or Calibration1).
%
% Pixel coordinates in the CSV/config are 0-based (OpenCV / web app). MATLAB
% pixels are 1-based, so 1 is added before using cameraParams.
%
% OUTPUT
%   validation_report.mat   per-measurement errors + statistics per method

function validate_measurements(csvPath)
scriptDir = fileparts(mfilename("fullpath"));
repoRoot = fileparts(scriptDir);
if nargin < 1
    csvPath = fullfile(scriptDir, "validation_measurements.csv");
end
tbl = readtable(csvPath, "TextType", "string");

if ismember("est_homography", tbl.Properties.VariableNames)
    fprintf("Web-export CSV: %s\n", csvPath);
    unit = "units";
    if ismember("unit", tbl.Properties.VariableNames) && height(tbl) > 0
        unit = string(tbl.unit(1));
    end
    gt = tbl.ground_truth;
    estH = tbl.est_homography;
    estP = nan(size(gt));
    if ismember("est_pinhole", tbl.Properties.VariableNames)
        estP = tbl.est_pinhole;
    end
else
    cfg = jsondecode(fileread(fullfile(scriptDir, "validation_config.json")));
    unit = string(cfg.unit);
    calPath = fullfile(scriptDir, "cameraParams.mat");
    if ~isfile(calPath)
        error("Run calibrate_from_folder or Calibration1 first (cameraParams.mat missing).");
    end
    loaded = load(calPath, "cameraParams");
    cameraParams = loaded.cameraParams;
    intr = cameraParams.Intrinsics;

    H = referenceHomography(cfg, cameraParams, repoRoot);

    zUnit = cfg.camera_distance_m / unitToMeters(unit);
    keep = tbl.ground_truth > 0;
    tbl = tbl(keep, :);
    gt = tbl.ground_truth;
    estH = nan(height(tbl), 1);
    estP = nan(height(tbl), 1);
    for i = 1:height(tbl)
        pts = [tbl.u1(i), tbl.v1(i); tbl.u2(i), tbl.v2(i)] + 1;
        und = undistortPoints(pts, cameraParams);
        estH(i) = norm(imageToWorld(H, und(2, :)) - imageToWorld(H, und(1, :)));
        % Pinhole: X = Z (u - cx) / fx, Y = Z (v - cy) / fy on a fronto-parallel plane.
        xy = zUnit * (und - intr.PrincipalPoint) ./ intr.FocalLength;
        estP(i) = norm(xy(2, :) - xy(1, :));
    end
end

statsH = errorStatistics(gt, estH);
statsP = errorStatistics(gt, estP);
fprintf("Valid measurements: %d (target 20)\n", statsH.n);
printStats("Homography (reference plane)", statsH, unit);
printStats("Pinhole (known distance)", statsP, unit);

results = table(gt, estH, estH - gt, estP, estP - gt, ...
    'VariableNames', ["ground_truth", "est_homography", "err_homography", "est_pinhole", "err_pinhole"]);
reportPath = fullfile(scriptDir, "validation_report.mat");
save(reportPath, "results", "statsH", "statsP", "unit");
fprintf("Saved %s\n", reportPath);
end

function H = referenceHomography(cfg, cameraParams, repoRoot)
% Homography mapping plane coordinates (X, Y) to undistorted pixels.
ref = cfg.reference;
switch string(ref.type)
    case "checkerboard"
        img = imread(fullfile(repoRoot, cfg.image_path));
        [imgPts, boardSize] = detectCheckerboardPoints(img);
        expected = reshape(ref.pattern, 1, []) + 1;
        if isempty(imgPts) || ~isequal(sort(boardSize), sort(expected))
            error("Checkerboard with %dx%d inner corners not found in %s.", ref.pattern(1), ref.pattern(2), cfg.image_path);
        end
        worldPts = generateCheckerboardPoints(boardSize, ref.square_size);
    case "rectangle"
        imgPts = double(ref.image_uv) + 1;   % jsondecode gives a 4x2 matrix
        w = ref.width; h = ref.height;
        worldPts = [0 0; w 0; w h; 0 h];
    otherwise
        error("Unknown reference type %s.", ref.type);
end
und = undistortPoints(imgPts, cameraParams);
tform = fitgeotform2d(worldPts, und, "projective");
if isprop(tform, "A")
    H = tform.A;
else
    H = tform.T';
end
end

function w = imageToWorld(H, uv)
q = H \ [uv, 1]';
w = (q(1:2) / q(3))';
end

function m = unitToMeters(unit)
switch string(unit)
    case "mm", m = 0.001;
    case "cm", m = 0.01;
    case "m",  m = 1;
    case "in", m = 0.0254;
    case "ft", m = 0.3048;
    otherwise, error("Unknown unit %s.", unit);
end
end

function s = errorStatistics(gt, est)
ok = isfinite(gt) & isfinite(est) & gt > 0;
e = est(ok) - gt(ok);
s.n = numel(e);
if s.n == 0
    return;
end
s.mae = mean(abs(e));
s.rmse = sqrt(mean(e .^ 2));
s.bias = mean(e);
s.std = 0;
s.ci95 = [s.bias, s.bias];
if s.n > 1
    s.std = std(e);
    if exist("tinv", "file")
        tcrit = tinv(0.975, s.n - 1);
    else
        tcrit = 1.96;
    end
    s.ci95 = s.bias + [-1, 1] * tcrit * s.std / sqrt(s.n);
end
s.mape_pct = 100 * mean(abs(e) ./ gt(ok));
s.max_abs = max(abs(e));
end

function printStats(name, s, unit)
fprintf("\n%s\n", name);
if s.n == 0
    fprintf("  no measurements\n");
    return;
end
fprintf("  n = %d\n  MAE  = %.4f %s\n  RMSE = %.4f %s\n  bias = %+.4f %s (95%% CI [%.4f, %.4f])\n", ...
    s.n, s.mae, unit, s.rmse, unit, s.bias, unit, s.ci95(1), s.ci95(2));
fprintf("  std  = %.4f %s\n  MAPE = %.2f %%\n  max |error| = %.4f %s\n", ...
    s.std, unit, s.mape_pct, s.max_abs, unit);
end
