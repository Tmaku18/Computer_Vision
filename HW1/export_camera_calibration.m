function export_camera_calibration(cameraParams, outputJsonPath)
%EXPORT_CAMERA_CALIBRATION  Write intrinsics to JSON for Python / documentation.
%
% README — HOW TO RUN
% ===================
% Run once after camera calibration (see Calibration1.m).
%
% From the MATLAB Command Window (HW1 folder on the path):
%
%   load("cameraParams.mat", "cameraParams");
%   export_camera_calibration(cameraParams, "camera_calibration.json");
%
% Or call automatically at the end of Calibration1.m (already wired there).
%
% OUTPUT
%   camera_calibration.json — camera_matrix (3×3), dist_coeffs, image_size,
%   world_units (e.g. "inches" from your checkerboard calibration).
%
% REQUIRES
%   MATLAB R2016b+ (jsonencode).

    if nargin < 2 || strlength(string(outputJsonPath)) == 0
        error("export_camera_calibration:NeedPath", ...
            "Second argument must be the output JSON file path.");
    end

    % MATLAB stores IntrinsicMatrix as transpose of standard K.
    K = cameraParams.IntrinsicMatrix';

    radial = cameraParams.RadialDistortion(:)';
    tang = cameraParams.TangentialDistortion(:)';
    dist = [radial, tang];
    % OpenCV order: k1, k2, p1, p2, k3
    if numel(dist) < 4
        dist = [dist, zeros(1, 4 - numel(dist))];
    end
    if numel(dist) == 4
        dist = [dist, 0];
    end
    dist = dist(1:5);

    imageSize = cameraParams.ImageSize; % [rows, cols]
    worldUnits = string(cameraParams.WorldUnits);

    payload = struct();
    payload.camera_matrix = K;
    payload.dist_coeffs = dist;
    payload.image_size = [imageSize(1), imageSize(2)];
    payload.world_units = char(worldUnits);
    payload.notes = "Exported from MATLAB cameraParameters. Use undistortPoints before homography.";

    jsonText = jsonencode(payload);
    fid = fopen(outputJsonPath, "w");
    if fid < 0
        error("export_camera_calibration:WriteFailed", "Could not open %s for writing.", outputJsonPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, "%s\n", jsonText);

    fprintf("Wrote calibration JSON: %s\n", outputJsonPath);
end
