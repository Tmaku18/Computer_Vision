Put four landscape photos of one flat textured object here. They are gitignored.

Same phone, 1× lens, and landscape orientation as the Module 2 calibration. About 20–40 degrees between views. Tape-measure the object's width (sets the scale) and height (the check), and note a rough camera position for each shot.

  .venv/bin/python HW4/structure_from_motion.py \
      --images HW4/sfm_images/v1.jpg HW4/sfm_images/v2.jpg HW4/sfm_images/v3.jpg HW4/sfm_images/v4.jpg \
      --clicks sfm_points.json --known-length 11 --unit in

Commit the downscaled files the script writes under HW4/results/sfm/, not these raw photos.
The synthetic poster from HW4/make_demo_results.py is only a stand-in so the hosted report has a reconstruction.
