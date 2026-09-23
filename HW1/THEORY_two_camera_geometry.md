# Theory — image coordinates of one point in two cameras

The same derivation is on the web app's **Theory** page. That page also has an interactive calculator that evaluates every equation below for any pose.

## Problem

A point $P = (X, Y, Z)$ is visible to camera 1, which is static, and to camera 2, which sits at some distance from camera 1 and is turned to an oblique orientation. We want the relationship between its image coordinates $\mathbf u_1 = (u_1, v_1)$ in camera 1 and $\mathbf u_2 = (u_2, v_2)$ in camera 2. Homogeneous pixels are written $\tilde{\mathbf u} = [u, v, 1]^\top$, and $\simeq$ means equal up to a nonzero scale.

## Assumptions and justification

1. **Pinhole cameras after lens-distortion removal.** Each pixel is first undistorted with the calibrated $(k_1, k_2)$ from Step 1. After that, a linear pinhole model fits the photos to about 2 px RMS.
2. **The world frame is camera 1's frame.** Camera 1 never moves, so its frame is a fixed reference: $R_1 = I,\ \mathbf t_1 = \mathbf 0$. This loses no generality, because any other world frame only adds one known rigid transform.
3. **Rigid, fixed cameras and a static scene** (or synchronized capture). The relative pose $(R, \mathbf t)$ is then constant.
4. **Known intrinsics** $K_1, K_2$ with zero skew, which holds for modern phone sensors. If the same phone is used twice, $K_1 = K_2$.
5. **$P$ is in front of both cameras** ($Z_1 > 0,\ Z_2 > 0$), **the baseline is nonzero** ($\mathbf t \ne \mathbf 0$), and $P$ is not on the baseline. These conditions are needed to recover depth.

## Static parameters, variables, and how to obtain them

| Symbol | Meaning | Type | How it is determined |
|---|---|---|---|
| $K_i = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix}$ | Intrinsics of camera $i$ (focal lengths, principal point, px) | static | Checkerboard calibration of each camera (Step 1, `cv2.calibrateCamera`) |
| $k_1, k_2$ | Radial distortion of each camera | static | Same calibration; used to undistort pixels before the linear model |
| $R$ (3 DOF) | Rotation from camera-1 axes to camera-2 axes (the oblique orientation) | static | Stereo calibration: both cameras see one checkerboard, and `solvePnP` gives $(R_{1b}, \mathbf t_{1b})$ and $(R_{2b}, \mathbf t_{2b})$. Then $R = R_{2b}R_{1b}^\top$; `cv2.stereoCalibrate` does this over many views. Alternatively, decompose $E$ (`findEssentialMat` + `recoverPose`). |
| $\mathbf t$ (3 DOF) | Translation of the camera-1 origin in the camera-2 frame. Camera 2's center is $\mathbf C_2 = -R^\top\mathbf t$, and the baseline is $B = \lVert\mathbf t\rVert$ | static | Same stereo calibration: $\mathbf t = \mathbf t_{2b} - R\,\mathbf t_{1b}$. From $E$ it is known only up to scale, which a tape-measured baseline $B$ fixes. |
| $H_\infty = K_2RK_1^{-1}$, $\tilde{\mathbf e}_2 = K_2\mathbf t$ | Infinite homography and epipole in image 2 | static (derived) | Computed from $K_1, K_2, R, \mathbf t$ |
| $E = [\mathbf t]_\times R$, $F = K_2^{-\top}EK_1^{-1}$ | Essential and fundamental matrices | static (derived) | Computed, or estimated directly from $\ge 8$ (F) or $\ge 5$ (E) point matches |
| $P = (X, Y, Z)$ | 3D point in the camera-1 frame | variable | Unknown; recovered by triangulation |
| $\mathbf u_1, \mathbf u_2$ | Pixel coordinates of $P$ in each image | variable | Measured (feature detection or clicking), then undistorted |
| $\lambda_1 = Z,\ \lambda_2 = Z_2$ | Depth of $P$ along each optical axis | variable | Unknown; $\lambda_1$ from triangulation, then $\lambda_2 = \mathbf r_3^\top P + t_z$ |

**Parameterizing the oblique pose.** Place camera 2's center at $\mathbf C_2$ (in camera-1 coordinates), then turn it by yaw $\theta$ about the vertical axis and pitch $\varphi$ about the horizontal axis. Camera 2's axes, in camera-1 coordinates, are the columns of $R_c = R_y(\theta)R_x(\varphi)$. A point transforms as $P_2 = R_c^\top(P - \mathbf C_2)$, so

$$R = R_c^\top, \qquad \mathbf t = -R\,\mathbf C_2 .$$

## Derivation

**Camera 1.** Using $K_1[\,I \mid \mathbf 0\,]$:

$$\lambda_1\tilde{\mathbf u}_1 = K_1P \iff u_1 = f_{x1}\frac{X}{Z} + c_{x1},\quad v_1 = f_{y1}\frac{Y}{Z} + c_{y1},\quad \lambda_1 = Z. \tag{1}$$

Inverting (1), the point lies on the back-projected ray of $\mathbf u_1$: $P = Z\,\mathbf x_1$, with normalized coordinates $\mathbf x_1 = K_1^{-1}\tilde{\mathbf u}_1$.

**Camera 2.** A rigid motion followed by projection:

$$P_2 = RP + \mathbf t, \qquad \lambda_2\tilde{\mathbf u}_2 = K_2(RP + \mathbf t). \tag{2}$$

**Image 1 to image 2.** Substituting $P = ZK_1^{-1}\tilde{\mathbf u}_1$ into (2):

$$\boxed{\lambda_2\tilde{\mathbf u}_2 = Z\,K_2RK_1^{-1}\tilde{\mathbf u}_1 + K_2\mathbf t = Z\,H_\infty\tilde{\mathbf u}_1 + \tilde{\mathbf e}_2} \tag{3}$$

In coordinates, with $\mathbf h_j^\top$ the rows of $H_\infty$ and $\tilde{\mathbf e}_2 = (e_x, e_y, e_z)$:

$$u_2 = \frac{Z\,\mathbf h_1^\top\tilde{\mathbf u}_1 + e_x}{Z\,\mathbf h_3^\top\tilde{\mathbf u}_1 + e_z},\qquad v_2 = \frac{Z\,\mathbf h_2^\top\tilde{\mathbf u}_1 + e_y}{Z\,\mathbf h_3^\top\tilde{\mathbf u}_1 + e_z},\qquad \lambda_2 = Z\,\mathbf h_3^\top\tilde{\mathbf u}_1 + e_z. \tag{4}$$

The camera-2 pixel is therefore fixed by the camera-1 pixel plus one unknown scalar, the depth $Z$. As $Z$ runs from 0 to $\infty$, $\mathbf u_2$ moves along a straight line from the epipole $\mathbf e_2$ ($Z\to 0$) to the vanishing point $H_\infty\tilde{\mathbf u}_1$ ($Z\to\infty$); that line is the epipolar line. The obliquity of camera 2 enters only through $R$, which rotates and skews the line through $H_\infty$. The baseline enters through $\tilde{\mathbf e}_2 = K_2\mathbf t$.

**Removing the depth: the epipolar constraint.** Write (3) in normalized coordinates, $\lambda_2\mathbf x_2 = Z\,R\mathbf x_1 + \mathbf t$, with $\mathbf x_2 = K_2^{-1}\tilde{\mathbf u}_2$. Taking the cross product with $\mathbf t$ removes the $\mathbf t$ term; taking the dot product with $\mathbf x_2$ then removes the left side:

$$\mathbf x_2^\top[\mathbf t]_\times R\,\mathbf x_1 = 0 \;\Longrightarrow\; \tilde{\mathbf u}_2^\top F\,\tilde{\mathbf u}_1 = 0,\qquad E = [\mathbf t]_\times R,\quad F = K_2^{-\top}EK_1^{-1},\quad [\mathbf t]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0\end{bmatrix}. \tag{5}$$

$E$ acts on normalized coordinates, and $F$ acts on pixels. The epipolar line in image 2 is $\mathbf l_2 = F\tilde{\mathbf u}_1$, and $\mathbf u_2$ must lie on it.

**Recovering $P$ (triangulation).** With both pixels measured, (3) gives three equations for the two unknown depths:

$$\underbrace{\begin{bmatrix} R\mathbf x_1 & -\mathbf x_2\end{bmatrix}}_{A}\begin{bmatrix} Z \\ \lambda_2\end{bmatrix} = -\mathbf t \;\Rightarrow\; \begin{bmatrix} Z \\ \lambda_2\end{bmatrix} = -(A^\top A)^{-1}A^\top\mathbf t,\qquad P = Z\,\mathbf x_1. \tag{6}$$

With noisy pixels the two rays do not meet exactly, and (6) is the least-squares solution; `cv2.triangulatePoints` uses the equivalent linear (DLT) form. The system is singular when $\mathbf t = \mathbf 0$ or when $P$ lies on the baseline, which is why assumption 5 is needed.

## Special cases

- **Pure rotation** ($\mathbf t = \mathbf 0$): $\tilde{\mathbf u}_2 \simeq H_\infty\tilde{\mathbf u}_1$. The mapping is a homography independent of depth, so depth cannot be recovered.
- **Points on a plane** $\mathbf n^\top P = d$: then $Z = d/(\mathbf n^\top\mathbf x_1)$ and $\tilde{\mathbf u}_2 \simeq K_2\left(R + \mathbf t\,\mathbf n^\top/d\right)K_1^{-1}\tilde{\mathbf u}_1$, the plane-induced homography. Step 2 uses the same idea, with the flat object plane itself as the "second view".
- **Parallel stereo** ($R = I,\ \mathbf t = (-B, 0, 0),\ K_1 = K_2$): (4) reduces to $v_2 = v_1$ and $u_2 = u_1 - f_xB/Z$, the familiar disparity–depth relation.

## Worked example (this project's calibration)

Both cameras use the Step 1 intrinsics: $f_x = 3729.9,\ f_y = 3729.8,\ c_x = 2861.9,\ c_y = 2112.2$ px. Camera 2 sits $B = 1$ m to the right, $\mathbf C_2 = (1, 0, 0)$ m, and is turned $\theta = -20^\circ$ (yaw) toward the scene. Take $P = (0.30, -0.20, 3.00)$ m.

1. Camera 1, from (1): $\mathbf u_1 = (3234.89,\ 1863.60)$ px and $\lambda_1 = 3.000$ m.
2. Pose: $R = R_y(-20^\circ)^\top = \begin{bmatrix} 0.9397 & 0 & 0.3420 \\ 0 & 1 & 0 \\ -0.3420 & 0 & 0.9397\end{bmatrix}$ and $\mathbf t = -R\mathbf C_2 = (-0.9397,\ 0,\ 0.3420)$ m.
3. $H_\infty = KRK^{-1} = \begin{bmatrix} 0.6773 & 0 & 2026.74 \\ -0.1937 & 1 & 426.92 \\ -9.17\times10^{-5} & 0 & 1.2021\end{bmatrix}$ and $\tilde{\mathbf e}_2 = K\mathbf t = (-2526.17,\ 722.43,\ 0.3420)$, so the epipole is at $(-7386.0,\ 2112.2)$ px, off-image to the left.
4. Equation (4) with $Z = 3$: $\mathbf u_2 = (3311.02,\ 1868.35)$ px and $\lambda_2 = 3.0585$ m. Direct projection $K(RP + \mathbf t)$ gives the same pixel, which checks (3).
5. $E = [\mathbf t]_\times R = \begin{bmatrix} 0 & -0.342 & 0 \\ 0 & 0 & 1 \\ 0 & -0.9397 & 0\end{bmatrix}$, and $\tilde{\mathbf u}_2^\top F\tilde{\mathbf u}_1 = 0$ to machine precision, which checks (5).
6. Triangulation (6) from $\mathbf u_1, \mathbf u_2$ returns $Z = 3.000$ m, $\lambda_2 = 3.0585$ m, and $P = (0.300, -0.200, 3.000)$ m, which checks (6).
