%% measure_planar_object_2d_demo.m
% README — SYNTHETIC DEMO (NO CALIBRATION FILE)
% ==============================================
%
% Verifies homography math with a simulated camera. Does not use your photos.
%
% HOW TO RUN (MATLAB)
%   cd("/Users/home/GSU/Computer_Vision/Computer_Vision/HW1")
%   measure_planar_object_2d_demo
%
% FOR REAL IMAGES
%   Run Calibration1.m, then measure_object_calibrated.m

trueW = 120; trueH = 75;
worldRef = [0 0; 200 0; 200 150; 0 150];
worldObj = [40 30; 40+trueW 30; 40+trueW 30+trueH; 40 30+trueH];

fx = 900; cx = 640; fy = 900; cy = 360;
K = [fx 0 cx; 0 fy cy; 0 0 1];
R = [0.95 0.05 -0.31; -0.10 0.98 -0.16; 0.29 0.19 0.94];
t = [-80; -60; 450];

imageRef = projectWorldDemo(K, R, t, [worldRef, zeros(4, 1)]);
imageObj = projectWorldDemo(K, R, t, [worldObj, zeros(4, 1)]);

res = measure_planar_object_2d(worldRef, imageRef, imageObj, "mm");
fprintf("True: %.1f x %.1f mm\n", trueW, trueH);
fprintf("Recovered: %.3f x %.3f mm\n", res.edgeLengths(1), res.edgeLengths(2));

function uv = projectWorldDemo(K, R, t, xyz)
    cam = R * xyz' + t;
    x = cam(1, :) ./ cam(3, :);
    y = cam(2, :) ./ cam(3, :);
    hom = K * [x; y; ones(1, size(xyz, 1))];
    uv = (hom(1:2, :) ./ hom(3, :))';
end
