function results = measure_planar_object_2d(referenceWorldXY, referenceImageUV, objectImageUV, worldUnits, cameraParams)
%MEASURE_PLANAR_OBJECT_2D  Real-world 2D sizes from perspective projection (planar H).
%
% README — CORE MEASUREMENT FUNCTION (library)
% ==============================================
%
% MODEL (object on plane Z = 0):
%   lambda * [u; v; 1] = H * [X; Y; 1]
%
% With cameraParams (optional 5th argument), referenceImageUV and objectImageUV
% may be **distorted** pixels; they are undistorted before H is estimated.
%
% TYPICAL USE
%   Run the executable script instead of calling this directly:
%     measure_object_calibrated.m
%
% DIRECT CALL (undistorted pixels already):
%   results = measure_planar_object_2d(refWorld, refUV, objUV, "inches");
%
% DIRECT CALL (distorted pixels + calibration):
%   results = measure_planar_object_2d(refWorld, refUV, objUV, "inches", cameraParams);
%
% REQUIRES
%   Computer Vision Toolbox (fitgeotform2d; undistortPoints if cameraParams given).

    if nargin < 4 || strlength(string(worldUnits)) == 0
        worldUnits = "mm";
    end

    referenceWorldXY = double(referenceWorldXY);
    referenceImageUV = double(referenceImageUV);
    objectImageUV = double(objectImageUV);

    if nargin >= 5 && ~isempty(cameraParams)
        referenceImageUV = undistortPoints(referenceImageUV, cameraParams);
        objectImageUV = undistortPoints(objectImageUV, cameraParams);
    end

    if size(referenceWorldXY, 2) ~= 2 || size(referenceImageUV, 2) ~= 2
        error("measure_planar_object_2d:BadSize", "Reference points must be Nx2.");
    end
    if size(referenceWorldXY, 1) < 4
        error("measure_planar_object_2d:Need4Pts", "Need at least 4 reference point pairs.");
    end

    % fitgeotform2d: moving (world) -> fixed (image).
    tform = fitgeotform2d(referenceWorldXY, referenceImageUV, "projective");
    H = tform.T';

    worldObj = imageToWorldPlane(H, objectImageUV);
    edgeLen = polygonEdgeLengths(worldObj, true);

    results = struct();
    results.worldUnits = string(worldUnits);
    results.worldXY = worldObj;
    results.edgeLengths = edgeLen;
    results.perimeter = sum(edgeLen);
    results.homography = H;

    fprintf("Edge lengths (%s): %s\n", worldUnits, mat2str(edgeLen, 4));
end

function worldXY = imageToWorldPlane(H, imageUV)
    n = size(imageUV, 1);
    uv1 = [imageUV, ones(n, 1)]';
    Hinv = inv(H);
    mapped = Hinv * uv1;
    worldXY = (mapped(1:2, :) ./ mapped(3, :))';
end

function lengths = polygonEdgeLengths(xy, closed)
    n = size(xy, 1);
    lengths = zeros(n - 1, 1);
    for i = 1:n - 1
        lengths(i) = norm(xy(i + 1, :) - xy(i, :));
    end
    if closed && n > 2
        lengths(end + 1) = norm(xy(1, :) - xy(end, :)); %#ok<AGROW>
    end
end
